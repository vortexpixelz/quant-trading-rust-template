//! Polymarket CLOB Quantitative Trading Engine
//!
//! Architecture:
//! 1. Connector fetches orderbook snapshots from CLOB API
//! 2. Orderbook module maintains local book state
//! 3. Signal generators compute trading signals (spread, imbalance, VPIN)
//! 4. Strategy engine combines signals into trade decisions
//! 5. (Future) Execution layer places orders via authenticated API

mod connectors;
mod orderbook;
mod signals;
mod strategies;

use std::time::Duration;
use tracing::{info, warn};
use tracing_subscriber::EnvFilter;

use connectors::polymarket::PolymarketClient;
use orderbook::book::OrderBook;
use signals::engine::SignalEngine;
use strategies::mean_reversion::MeanReversionStrategy;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .with_target(false)
        .init();

    info!("Polymarket CLOB Trading Engine starting");

    // Example: monitor a market's orderbook and generate signals
    let client = PolymarketClient::new(None)?;

    // Discover active markets
    let markets = client.get_sampling_markets().await?;
    info!(market_count = markets.len(), "discovered active markets");

    if markets.is_empty() {
        warn!("no active markets found");
        return Ok(());
    }

    // Pick first market's YES token for demonstration
    let market = &markets[0];
    let token_id = market
        .tokens
        .first()
        .map(|t| t.token_id.clone())
        .unwrap_or_default();

    info!(
        question = %market.question,
        token = &token_id[..20.min(token_id.len())],
        "monitoring market"
    );

    // Initialize components
    let mut book = OrderBook::new(token_id.clone());
    let mut signal_engine = SignalEngine::new(50); // 50-period lookback
    let strategy = MeanReversionStrategy::new(
        0.02,  // entry threshold: 2% deviation from fair value
        0.005, // exit threshold: 0.5% mean reversion
        1000.0, // max position size in USDC
    );

    // Main loop: poll orderbook and generate signals
    loop {
        match client.get_book(&token_id).await {
            Ok(snapshot) => {
                book.update_from_snapshot(&snapshot);

                let mid = book.mid_price();
                let spread = book.spread();
                let imbalance = book.imbalance();
                let bid_depth = book.total_bid_depth();
                let ask_depth = book.total_ask_depth();

                // Feed to signal engine
                signal_engine.push(mid, spread, imbalance, bid_depth, ask_depth);

                // Generate signals
                if let Some(sigs) = signal_engine.compute() {
                    let decision = strategy.evaluate(&sigs, &book);

                    info!(
                        mid = %format!("{:.4}", mid),
                        spread = %format!("{:.4}", spread),
                        imbalance = %format!("{:.4}", imbalance),
                        zscore = %format!("{:.3}", sigs.price_zscore),
                        vpin = %format!("{:.3}", sigs.vpin),
                        signal = ?decision,
                        "tick"
                    );
                }
            }
            Err(e) => {
                warn!(error = %e, "failed to fetch orderbook");
            }
        }

        tokio::time::sleep(Duration::from_secs(5)).await;
    }
}
