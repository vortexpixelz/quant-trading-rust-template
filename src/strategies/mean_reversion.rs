//! Mean-reversion strategy for prediction markets.
//!
//! Core thesis: prediction market prices tend to revert to "fair value"
//! after short-term dislocations caused by uninformed flow. When the
//! price Z-score exceeds a threshold AND VPIN is low (indicating the
//! move is noise, not signal), we fade the move.
//!
//! Guard rails:
//! - Skip entry when VPIN is high (informed flow — don't fade smart money)
//! - Scale position by spread (wider spread = less confidence)
//! - Maximum position size cap

use crate::orderbook::book::OrderBook;
use crate::signals::engine::Signals;

/// Trade decision output.
#[derive(Debug, Clone)]
pub enum TradeDecision {
    /// No action.
    Hold,
    /// Buy at market (fade downward dislocation).
    Buy { size: f64, reason: String },
    /// Sell at market (fade upward dislocation).
    Sell { size: f64, reason: String },
}

/// Mean-reversion strategy parameters.
pub struct MeanReversionStrategy {
    /// Z-score threshold to enter a position (e.g., 2.0 = 2 sigma).
    entry_threshold: f64,
    /// Maximum position size in USDC.
    max_position: f64,
    /// VPIN threshold above which we skip entry (smart money guard).
    vpin_guard: f64,
    /// Minimum spread to trade (avoid illiquid moments).
    min_spread: f64,
}

impl MeanReversionStrategy {
    pub fn new(entry_threshold: f64, max_position: f64) -> Self {
        Self {
            entry_threshold,
            max_position,
            vpin_guard: 0.7,
            min_spread: 0.001,
        }
    }

    /// Evaluate current signals and produce a trade decision.
    pub fn evaluate(&self, signals: &Signals, book: &OrderBook) -> TradeDecision {
        // Guard: skip if spread is too tight (likely stale book)
        if signals.spread < self.min_spread {
            return TradeDecision::Hold;
        }

        // Guard: skip if VPIN is high (informed flow — don't fade it)
        if signals.vpin > self.vpin_guard {
            return TradeDecision::Hold;
        }

        let z = signals.price_zscore;

        // Entry: price deviated significantly from mean
        if z < -self.entry_threshold {
            // Price is unusually LOW → buy (expect reversion up)
            let confidence = (z.abs() - self.entry_threshold) / self.entry_threshold;
            let spread_penalty = 1.0 - (signals.spread * 50.0).min(0.8); // wider spread = smaller size
            let size = (self.max_position * confidence.min(1.0) * spread_penalty).max(5.0);

            // Check we can actually fill at reasonable price
            let impact = book.buy_impact(size);
            if impact > signals.mid_price * 1.05 {
                return TradeDecision::Hold; // too much slippage
            }

            return TradeDecision::Buy {
                size,
                reason: format!(
                    "z={:.2} < -{:.2}, vpin={:.2}, spread={:.4}",
                    z, self.entry_threshold, signals.vpin, signals.spread
                ),
            };
        }

        if z > self.entry_threshold {
            // Price is unusually HIGH → sell (expect reversion down)
            let confidence = (z.abs() - self.entry_threshold) / self.entry_threshold;
            let spread_penalty = 1.0 - (signals.spread * 50.0).min(0.8);
            let size = (self.max_position * confidence.min(1.0) * spread_penalty).max(5.0);

            let impact = book.sell_impact(size);
            if impact < signals.mid_price * 0.95 {
                return TradeDecision::Hold;
            }

            return TradeDecision::Sell {
                size,
                reason: format!(
                    "z={:.2} > {:.2}, vpin={:.2}, spread={:.4}",
                    z, self.entry_threshold, signals.vpin, signals.spread
                ),
            };
        }

        TradeDecision::Hold
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::connectors::polymarket::{OrderBookSnapshot, PriceLevel};
    use crate::orderbook::book::OrderBook;
    use crate::signals::engine::Signals;

    fn make_book() -> OrderBook {
        let snap = OrderBookSnapshot {
            market: "0xtest".into(),
            asset_id: "123".into(),
            timestamp: "1000".into(),
            hash: "abc".into(),
            bids: vec![
                PriceLevel {
                    price: "0.50".into(),
                    size: "1000".into(),
                },
                PriceLevel {
                    price: "0.48".into(),
                    size: "2000".into(),
                },
            ],
            asks: vec![
                PriceLevel {
                    price: "0.52".into(),
                    size: "1000".into(),
                },
                PriceLevel {
                    price: "0.55".into(),
                    size: "2000".into(),
                },
            ],
            min_order_size: "5".into(),
            tick_size: "0.01".into(),
            neg_risk: false,
            last_trade_price: "0.51".into(),
        };
        let mut book = OrderBook::new("123".into());
        book.update_from_snapshot(&snap);
        book
    }

    #[test]
    fn test_hold_when_zscore_normal() {
        let strategy = MeanReversionStrategy::new(2.0, 1000.0);
        let signals = Signals {
            price_zscore: 0.5,
            spread_zscore: 0.0,
            imbalance_ema: 0.0,
            imbalance_momentum: 0.0,
            vpin: 0.3,
            mid_price: 0.51,
            spread: 0.02,
            imbalance: 0.0,
            bid_depth: 3000.0,
            ask_depth: 3000.0,
        };
        let book = make_book();
        assert!(matches!(
            strategy.evaluate(&signals, &book),
            TradeDecision::Hold
        ));
    }

    #[test]
    fn test_buy_when_price_low() {
        let strategy = MeanReversionStrategy::new(2.0, 1000.0);
        let signals = Signals {
            price_zscore: -3.0,
            spread_zscore: 0.0,
            imbalance_ema: 0.0,
            imbalance_momentum: 0.0,
            vpin: 0.3,
            mid_price: 0.51,
            spread: 0.02,
            imbalance: 0.0,
            bid_depth: 3000.0,
            ask_depth: 3000.0,
        };
        let book = make_book();
        assert!(matches!(
            strategy.evaluate(&signals, &book),
            TradeDecision::Buy { .. }
        ));
    }

    #[test]
    fn test_hold_when_vpin_high() {
        let strategy = MeanReversionStrategy::new(2.0, 1000.0);
        let signals = Signals {
            price_zscore: -3.0,
            spread_zscore: 0.0,
            imbalance_ema: 0.0,
            imbalance_momentum: 0.0,
            vpin: 0.85, // high VPIN — smart money guard
            mid_price: 0.51,
            spread: 0.02,
            imbalance: 0.0,
            bid_depth: 3000.0,
            ask_depth: 3000.0,
        };
        let book = make_book();
        assert!(matches!(
            strategy.evaluate(&signals, &book),
            TradeDecision::Hold
        ));
    }

    #[test]
    fn test_sell_when_price_high() {
        let strategy = MeanReversionStrategy::new(2.0, 1000.0);
        let signals = Signals {
            price_zscore: 3.5,
            spread_zscore: 0.0,
            imbalance_ema: 0.0,
            imbalance_momentum: 0.0,
            vpin: 0.2,
            mid_price: 0.51,
            spread: 0.02,
            imbalance: 0.0,
            bid_depth: 3000.0,
            ask_depth: 3000.0,
        };
        let book = make_book();
        assert!(matches!(
            strategy.evaluate(&signals, &book),
            TradeDecision::Sell { .. }
        ));
    }
}
