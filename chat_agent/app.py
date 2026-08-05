"""TradingAgents Chat Agent — Streamlit web app.

A conversational interface where users enter a stock ticker and receive
a full multi-agent trading analysis report.
"""

import contextlib
import datetime
import json
import os
import re
import sys
from pathlib import Path
from urllib.parse import unquote

import streamlit as st
from stockstats import wrap
from streamlit.components.v1 import html as components_html

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tradingagents.dataflows.stockstats_utils import load_ohlcv
from tradingagents.dataflows.symbol_utils import NoMarketDataError
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph

CRYPTO_SUFFIXES = ("-USD", "-USDT", "-USDC", "-BTC", "-ETH")

# The decision-making agents (Trader, Portfolio Manager, Research Manager)
# render their structured output to markdown with a stable "**Label**: value"
# shape (see tradingagents/agents/schemas.py) — these values are extracted
# straight from that markdown rather than re-running the LLM, so the charts
# below always reflect exactly what the report says.
_FIELD_RE_CACHE: dict[str, re.Pattern] = {}


def _extract_field(text: str, label: str) -> str | None:
    """Pull "**Label**: value" out of a rendered agent report, or None."""
    if not text:
        return None
    pattern = _FIELD_RE_CACHE.setdefault(
        label, re.compile(rf"\*\*{re.escape(label)}\*\*:\s*(.+)")
    )
    match = pattern.search(text)
    return match.group(1).strip() if match else None


def _extract_float_field(text: str, label: str) -> float | None:
    value = _extract_field(text, label)
    if not value:
        return None
    try:
        return float(value.replace("$", "").replace(",", "").split()[0])
    except ValueError:
        return None


def _extract_sentiment(sentiment_report: str) -> tuple[str, float, str] | None:
    """Parse the SentimentReport header rendered by render_sentiment_report."""
    if not sentiment_report:
        return None
    match = re.search(
        r"\*\*Overall Sentiment:\*\*\s*\*\*([^*]+)\*\*\s*\(Score:\s*([\d.]+)/10\)",
        sentiment_report,
    )
    if not match:
        return None
    band, score = match.group(1).strip(), float(match.group(2))
    confidence_match = re.search(r"\*\*Confidence:\*\*\s*(\w+)", sentiment_report)
    confidence = confidence_match.group(1) if confidence_match else "unknown"
    return band, score, confidence


def detect_asset_type(ticker: str) -> str:
    if ticker.upper().endswith(CRYPTO_SUFFIXES):
        return "crypto"
    return "stock"


def _set_browser_cookie(name: str, value: str, days: int = 90) -> None:
    """Persist a value in a cookie on the visitor's own browser (not the server).

    Distinct from the ``api_key`` in ``st.session_state``: session state is
    wiped whenever the session ends (tab closed, app redeployed), which is
    exactly why the same key had to be re-entered on every visit. A cookie
    survives that, while still never touching the server's environment and
    never being visible to any other visitor — same security property as
    session state, just persisted client-side instead of in-memory.

    Streamlit renders each component in its own iframe, so this reaches into
    ``window.parent.document`` to set the cookie on the actual top-level page.
    """
    max_age = days * 24 * 60 * 60
    components_html(
        f"""
        <script>
        (function() {{
            const secure = window.parent.location.protocol === 'https:' ? '; Secure' : '';
            window.parent.document.cookie =
                {json.dumps(name)} + '=' + encodeURIComponent({json.dumps(value)}) +
                '; path=/; max-age={max_age}; SameSite=Lax' + secure;
        }})();
        </script>
        """,
        height=0,
        width=0,
    )


def _clear_browser_cookie(name: str) -> None:
    components_html(
        f"""
        <script>
        window.parent.document.cookie = {json.dumps(name)} + '=; path=/; max-age=0; SameSite=Lax';
        </script>
        """,
        height=0,
        width=0,
    )


def _server_default_key(env_name: str) -> str | None:
    """A key the app owner configured once (Streamlit Cloud's Secrets panel),
    used automatically for every visitor so nobody has to bring their own.

    Streamlit promotes top-level st.secrets entries into os.environ on load,
    so this also transparently covers a plain env var set the same way
    outside Streamlit Cloud (e.g. local dev, another host) — one check
    handles both, and avoids st.secrets raising when no secrets.toml exists
    at all (the common case for a visitor's own local checkout).
    """
    return os.environ.get(env_name) or None


def _render_api_key_input(provider: str, api_key_label: str) -> str | None:
    """Text input for a visitor's own key, persisted via a per-provider cookie.

    Returns the entered key, or None. Reused both when a key is required
    (no server default configured) and when it's an optional override (a
    server default exists but a visitor wants to use a different provider
    or a higher-tier key of their own).
    """
    key_cache = st.session_state.setdefault("_local_key_cache", {})
    cookie_name = f"ta_key_{provider}"
    widget_key = f"api_key_input_{provider}"

    if widget_key not in st.session_state:
        if cookie_name not in key_cache:
            key_cache[cookie_name] = unquote(st.context.cookies.get(cookie_name, ""))
        st.session_state[widget_key] = key_cache[cookie_name]

    def _forget_saved_key(widget_key=widget_key, cookie_name=cookie_name):
        # Runs as an on_click callback (before the widget below is
        # re-instantiated), which is the only safe time to overwrite a
        # widget-bound session_state entry.
        st.session_state[widget_key] = ""
        st.session_state.setdefault("_local_key_cache", {})[cookie_name] = ""
        st.session_state["_pending_cookie_clear"] = cookie_name

    api_key = st.text_input(
        f"{api_key_label}",
        type="password",
        key=widget_key,
        help=(
            "Saved as a cookie on this device only — never on the server, "
            "never visible to other visitors — so you don't have to "
            "re-enter it every time you open the app."
        ),
    )
    if api_key:
        st.button("Forget saved key", key=f"forget_{provider}", on_click=_forget_saved_key)

    pending_clear = st.session_state.pop("_pending_cookie_clear", None)
    if pending_clear:
        _clear_browser_cookie(pending_clear)
    elif api_key:
        key_cache[cookie_name] = api_key
        _set_browser_cookie(cookie_name, api_key)

    return api_key


def format_report(final_state: dict, ticker: str) -> str:
    """Build a markdown report from the final graph state."""
    sections = []
    sections.append(f"# Trading Analysis Report: {ticker}")
    sections.append(
        f"*Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*"
    )

    analyst_parts = []
    for key, label in [
        ("market_report", "Market Analyst"),
        ("sentiment_report", "Sentiment Analyst"),
        ("news_report", "News Analyst"),
        ("fundamentals_report", "Fundamentals Analyst"),
    ]:
        if final_state.get(key):
            analyst_parts.append(f"### {label}\n{final_state[key]}")
    if analyst_parts:
        sections.append("## I. Analyst Team Reports\n\n" + "\n\n".join(analyst_parts))

    if final_state.get("investment_debate_state"):
        debate = final_state["investment_debate_state"]
        research_parts = []
        if debate.get("bull_history"):
            research_parts.append(f"### Bull Researcher\n{debate['bull_history']}")
        if debate.get("bear_history"):
            research_parts.append(f"### Bear Researcher\n{debate['bear_history']}")
        if debate.get("judge_decision"):
            research_parts.append(
                f"### Research Manager\n{debate['judge_decision']}"
            )
        if research_parts:
            sections.append(
                "## II. Research Team Decision\n\n" + "\n\n".join(research_parts)
            )

    if final_state.get("investment_plan"):
        sections.append(
            f"## III. Investment Plan\n\n{final_state['investment_plan']}"
        )

    if final_state.get("trader_investment_plan"):
        sections.append(
            f"## IV. Trading Team Plan\n\n### Trader\n{final_state['trader_investment_plan']}"
        )

    if final_state.get("risk_debate_state"):
        risk = final_state["risk_debate_state"]
        risk_parts = []
        if risk.get("aggressive_history"):
            risk_parts.append(
                f"### Aggressive Analyst\n{risk['aggressive_history']}"
            )
        if risk.get("conservative_history"):
            risk_parts.append(
                f"### Conservative Analyst\n{risk['conservative_history']}"
            )
        if risk.get("neutral_history"):
            risk_parts.append(f"### Neutral Analyst\n{risk['neutral_history']}")
        if risk_parts:
            sections.append(
                "## V. Risk Management Team\n\n" + "\n\n".join(risk_parts)
            )
        if risk.get("judge_decision"):
            sections.append(
                f"## VI. Portfolio Manager Decision\n\n{risk['judge_decision']}"
            )

    if final_state.get("final_trade_decision"):
        sections.append(
            f"## Final Decision\n\n{final_state['final_trade_decision']}"
        )

    return "\n\n".join(sections)


def render_price_chart(ticker: str, trade_date: str, final_state: dict):
    """Price chart with the trader's entry/stop and the PM's price target overlaid.

    Returns the underlying OHLCV frame (or None if it couldn't be fetched) so
    the caller can reuse it for the indicator dashboard without a second fetch.
    """
    entry_price = _extract_float_field(final_state.get("trader_investment_plan", ""), "Entry Price")
    stop_loss = _extract_float_field(final_state.get("trader_investment_plan", ""), "Stop Loss")
    price_target = _extract_float_field(final_state.get("final_trade_decision", ""), "Price Target")

    try:
        ohlcv = load_ohlcv(ticker, trade_date)
    except NoMarketDataError:
        return None
    if ohlcv is None or ohlcv.empty:
        return None

    st.subheader("📊 Price Chart")
    chart_df = ohlcv.tail(120).set_index("Date")[["Close"]].rename(columns={"Close": "Price"})
    if entry_price:
        chart_df["Entry"] = entry_price
    if stop_loss:
        chart_df["Stop Loss"] = stop_loss
    if price_target:
        chart_df["Target"] = price_target
    st.line_chart(chart_df)

    cols = st.columns(3)
    cols[0].metric("Entry", f"${entry_price:,.2f}" if entry_price else "—")
    cols[1].metric("Stop Loss", f"${stop_loss:,.2f}" if stop_loss else "—")
    cols[2].metric("Target", f"${price_target:,.2f}" if price_target else "—")

    return ohlcv


def render_indicator_dashboard(ohlcv) -> None:
    """RSI / MACD / moving-average dashboard computed from the same OHLCV frame."""
    if ohlcv is None or ohlcv.empty:
        return

    df = wrap(ohlcv.copy())
    for col in ("rsi", "macd", "macds", "macdh", "close_50_sma", "close_10_ema"):
        df[col]  # noqa: B018 - triggers stockstats' lazy indicator computation
    tail = df.tail(90)
    latest = tail.iloc[-1]

    st.subheader("📈 Indicator Dashboard")
    cols = st.columns(3)
    cols[0].metric("RSI (14)", f"{latest['rsi']:.1f}")
    cols[1].metric("MACD", f"{latest['macd']:.3f}")
    cols[2].metric("MACD Signal", f"{latest['macds']:.3f}")

    price_cols = tail.set_index("Date")[["close", "close_50_sma", "close_10_ema"]].rename(
        columns={"close": "Close", "close_50_sma": "50-day SMA", "close_10_ema": "10-day EMA"}
    )
    st.line_chart(price_cols)
    st.line_chart(tail.set_index("Date")[["rsi"]].rename(columns={"rsi": "RSI (14)"}))
    st.bar_chart(tail.set_index("Date")[["macdh"]].rename(columns={"macdh": "MACD Histogram"}))


def render_sentiment_gauge(final_state: dict) -> None:
    parsed = _extract_sentiment(final_state.get("sentiment_report", ""))
    if not parsed:
        return
    band, score, confidence = parsed

    st.subheader("🎯 Sentiment Gauge")
    cols = st.columns([2, 1])
    cols[0].progress(score / 10, text=f"{band} — {score:.1f}/10")
    cols[1].metric("Confidence", confidence.capitalize())


def render_decision_summary(final_state: dict, decision: str) -> None:
    """Bull-vs-bear debate balance plus how the Research and Portfolio Managers ruled."""
    debate = final_state.get("investment_debate_state") or {}
    bull_words = len(debate.get("bull_history", "").split())
    bear_words = len(debate.get("bear_history", "").split())

    st.subheader("⚖️ Decision Summary")
    if bull_words or bear_words:
        st.caption("Debate balance (words argued by each side)")
        st.bar_chart({"Bull": bull_words, "Bear": bear_words})

    research_call = _extract_field(final_state.get("investment_plan", ""), "Recommendation")
    cols = st.columns(3)
    cols[0].metric("Research Manager", research_call or "—")
    cols[1].metric("Portfolio Manager", _extract_field(final_state.get("final_trade_decision", ""), "Rating") or "—")
    cols[2].metric("Final Signal", decision or "—")

    position_sizing = _extract_field(final_state.get("trader_investment_plan", ""), "Position Sizing")
    if position_sizing:
        st.caption(f"**Suggested position size:** {position_sizing}")


def render_visuals(ticker: str, trade_date: str, final_state: dict, decision: str) -> None:
    """Render the report's visual layer: price/indicator charts, sentiment, decision summary.

    Each block is independently best-effort: the price chart needs a live
    OHLCV fetch and can fail (unsupported symbol, network hiccup), but that
    must not hide the sentiment gauge or decision summary, which only need
    data already in ``final_state``.
    """
    ohlcv = None
    with contextlib.suppress(Exception):
        ohlcv = render_price_chart(ticker, trade_date, final_state)
    with contextlib.suppress(Exception):
        render_indicator_dashboard(ohlcv)
    with contextlib.suppress(Exception):
        render_sentiment_gauge(final_state)
    with contextlib.suppress(Exception):
        render_decision_summary(final_state, decision)


def run_analysis(ticker: str, trade_date: str, config: dict) -> tuple:
    """Run the TradingAgents pipeline and return the final state."""
    asset_type = detect_asset_type(ticker)

    analysts = ["market", "social", "news", "fundamentals"]
    if asset_type == "crypto":
        analysts = [a for a in analysts if a != "fundamentals"]

    graph = TradingAgentsGraph(
        selected_analysts=analysts,
        config=config,
        debug=False,
    )
    final_state, decision = graph.propagate(ticker, trade_date, asset_type=asset_type)
    return final_state, decision


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="TradingAgents Chat",
    page_icon="📈",
    layout="wide",
)

st.title("📈 TradingAgents Chat")
st.caption(
    "Enter a stock ticker to get a full AI-powered trading analysis report."
)

# Sidebar: configuration
with st.sidebar:
    st.header("Configuration")

    llm_provider = st.selectbox(
        "LLM Provider",
        ["openrouter", "groq", "openai", "anthropic", "google", "deepseek", "ollama"],
        index=0,
        help=(
            "OpenRouter is first — once the app owner sets a shared key, it "
            "works with no setup on your end and gives access to a wide "
            "range of models. Other providers need your own key below."
        ),
    )

    provider_models = {
        "openai": ("gpt-5.5", "gpt-5.4-mini"),
        "anthropic": ("claude-sonnet-4-6", "claude-haiku-4-5-20251001"),
        "google": ("gemini-2.5-pro", "gemini-2.5-flash"),
        "deepseek": ("deepseek-chat", "deepseek-chat"),
        # Both deep and quick default to the same Groq model: this app's
        # prompts (tool lists, indicator descriptions, growing conversation
        # history) routinely exceed llama-3.1-8b-instant's free-tier
        # per-minute token budget on Groq (a hard per-request ceiling — no
        # amount of retrying fixes a request that's simply too large), which
        # surfaced as HTTP 413 "Request too large" failures. 70b-versatile
        # has enough headroom for this app's actual prompt sizes.
        "groq": ("llama-3.3-70b-versatile", "llama-3.3-70b-versatile"),
        "openrouter": ("anthropic/claude-sonnet-4.6", "openai/gpt-5.4-mini"),
        "ollama": ("glm-4.7-flash:latest", "qwen3:latest"),
    }
    default_deep, default_quick = provider_models.get(
        llm_provider, ("gpt-5.5", "gpt-5.4-mini")
    )

    deep_model = st.text_input("Deep thinking model", value=default_deep)
    quick_model = st.text_input("Quick thinking model", value=default_quick)
    debate_rounds = st.slider(
        "Debate rounds",
        1,
        2,
        2,
        help=(
            "1 = opening statements only (Bull, Bear, then the Research "
            "Manager's decision). 2 = opening statements plus a rebuttal round. "
            "More rounds tend to repeat earlier points rather than add new "
            "information, so this is capped at 2."
        ),
    )

    backend_url = None
    if llm_provider == "ollama":
        backend_url = st.text_input(
            "Ollama base URL",
            value="http://localhost:11434/v1",
            help=(
                "Streamlit Cloud cannot reach a localhost Ollama server — "
                "point this at a publicly reachable Ollama endpoint."
            ),
        )

    api_key_label = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GOOGLE_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "groq": "GROQ_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
    }.get(llm_provider)

    server_key = _server_default_key(api_key_label) if api_key_label else None

    api_key = None
    if server_key:
        st.success("Using a shared key — no setup needed.", icon="✅")
        with st.expander("Use your own key instead"):
            api_key = _render_api_key_input(llm_provider, api_key_label)
    elif api_key_label:
        api_key = _render_api_key_input(llm_provider, api_key_label)
    else:
        st.caption("No API key required for this provider.")

    if api_key:
        st.session_state["api_key"] = api_key
    else:
        st.session_state.pop("api_key", None)

    st.divider()
    st.markdown(
        "**Note:** Analysis takes 2-5 minutes per ticker depending on "
        "model speed and debate rounds."
    )

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Welcome! I can analyze any stock or crypto for you.\n\n"
                "Just type a **ticker symbol** (e.g. `NVDA`, `AAPL`, `BTC-USD`) "
                "and I'll run a full multi-agent analysis including:\n"
                "- Market & technical analysis\n"
                "- Sentiment analysis\n"
                "- News analysis\n"
                "- Fundamentals analysis\n"
                "- Bull vs Bear debate\n"
                "- Risk assessment\n"
                "- Final portfolio decision\n\n"
                "You can optionally include a date like `NVDA 2026-06-25`."
            ),
        }
    ]

if "running" not in st.session_state:
    st.session_state.running = False

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input(
    "Enter a ticker (e.g. NVDA, AAPL, BTC-USD)",
    disabled=st.session_state.running,
)

if user_input and not st.session_state.running:
    parts = user_input.strip().split()
    ticker = parts[0].upper()
    trade_date = parts[1] if len(parts) > 1 else datetime.date.today().strftime("%Y-%m-%d")

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    asset_type = detect_asset_type(ticker)
    asset_label = "crypto asset" if asset_type == "crypto" else "stock"

    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        status_placeholder.markdown(
            f"🔄 **Analyzing {ticker}** ({asset_label}) for **{trade_date}**...\n\n"
            f"This will take a few minutes. The multi-agent pipeline is running:\n"
            f"1. Analyst team gathering data\n"
            f"2. Bull vs Bear research debate\n"
            f"3. Trader formulating a plan\n"
            f"4. Risk management team evaluation\n"
            f"5. Portfolio manager final decision"
        )

        config = DEFAULT_CONFIG.copy()
        config["llm_provider"] = llm_provider
        config["deep_think_llm"] = deep_model
        config["quick_think_llm"] = quick_model
        config["max_debate_rounds"] = debate_rounds
        config["max_risk_discuss_rounds"] = debate_rounds
        if backend_url:
            config["backend_url"] = backend_url
        session_api_key = st.session_state.get("api_key")
        if session_api_key:
            config["api_key"] = session_api_key

        st.session_state.running = True
        try:
            final_state, decision = run_analysis(ticker, trade_date, config)
            report = format_report(final_state, ticker)

            status_placeholder.empty()

            # Visuals are a bonus on top of the text report — a charting
            # failure (e.g. no OHLCV data for this symbol) must not hide an
            # otherwise-successful analysis.
            with contextlib.suppress(Exception):
                render_visuals(ticker, trade_date, final_state, decision)

            st.markdown(report)

            st.success(f"**Signal: {decision}**")

            st.session_state.messages.append(
                {"role": "assistant", "content": report + f"\n\n**Signal: {decision}**"}
            )

            report_bytes = report.encode("utf-8")
            st.download_button(
                label="📥 Download Report",
                data=report_bytes,
                file_name=f"{ticker}_{trade_date}_report.md",
                mime="text/markdown",
            )

        except Exception as e:
            status_placeholder.empty()
            error_msg = f"❌ **Analysis failed:** {e}"
            st.error(error_msg)
            st.session_state.messages.append(
                {"role": "assistant", "content": error_msg}
            )
        finally:
            st.session_state.running = False
