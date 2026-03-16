//! Local orderbook representation with computed analytics.
//!
//! Maintains a sorted book from CLOB API snapshots and provides
//! real-time metrics: mid-price, spread, imbalance, depth.

use crate::connectors::polymarket::{OrderBookSnapshot, PriceLevel};

/// A single price level in the local book.
#[derive(Debug, Clone)]
pub struct Level {
    pub price: f64,
    pub size: f64,
}

impl Level {
    fn from_api(pl: &PriceLevel) -> Self {
        Self {
            price: pl.price.parse().unwrap_or(0.0),
            size: pl.size.parse().unwrap_or(0.0),
        }
    }
}

/// Local orderbook state for a single token.
#[derive(Debug)]
pub struct OrderBook {
    pub token_id: String,
    pub bids: Vec<Level>, // sorted best (highest) first
    pub asks: Vec<Level>, // sorted best (lowest) first
    pub last_update_ms: u64,
    pub hash: String,
    pub last_trade_price: f64,
}

impl OrderBook {
    pub fn new(token_id: String) -> Self {
        Self {
            token_id,
            bids: Vec::new(),
            asks: Vec::new(),
            last_update_ms: 0,
            hash: String::new(),
            last_trade_price: 0.0,
        }
    }

    /// Replace local book state with a fresh API snapshot.
    pub fn update_from_snapshot(&mut self, snap: &OrderBookSnapshot) {
        self.bids = snap.bids.iter().map(Level::from_api).collect();
        self.asks = snap.asks.iter().map(Level::from_api).collect();

        // Ensure sorting: bids descending by price, asks ascending
        self.bids
            .sort_by(|a, b| b.price.partial_cmp(&a.price).unwrap());
        self.asks
            .sort_by(|a, b| a.price.partial_cmp(&b.price).unwrap());

        self.last_update_ms = snap.timestamp.parse().unwrap_or(0);
        self.hash = snap.hash.clone();
        self.last_trade_price = snap.last_trade_price.parse().unwrap_or(0.0);
    }

    /// Best bid price (highest buy order).
    pub fn best_bid(&self) -> f64 {
        self.bids.first().map(|l| l.price).unwrap_or(0.0)
    }

    /// Best ask price (lowest sell order).
    pub fn best_ask(&self) -> f64 {
        self.asks.first().map(|l| l.price).unwrap_or(1.0)
    }

    /// Mid-price: (best_bid + best_ask) / 2.
    pub fn mid_price(&self) -> f64 {
        (self.best_bid() + self.best_ask()) / 2.0
    }

    /// Bid-ask spread in absolute terms.
    pub fn spread(&self) -> f64 {
        self.best_ask() - self.best_bid()
    }

    /// Spread in basis points relative to mid-price.
    pub fn spread_bps(&self) -> f64 {
        let mid = self.mid_price();
        if mid > 0.0 {
            self.spread() / mid * 10_000.0
        } else {
            0.0
        }
    }

    /// Total bid-side liquidity (sum of all bid sizes).
    pub fn total_bid_depth(&self) -> f64 {
        self.bids.iter().map(|l| l.size).sum()
    }

    /// Total ask-side liquidity (sum of all ask sizes).
    pub fn total_ask_depth(&self) -> f64 {
        self.asks.iter().map(|l| l.size).sum()
    }

    /// Order book imbalance: (bid_depth - ask_depth) / (bid_depth + ask_depth).
    /// Positive = more buy pressure, negative = more sell pressure.
    pub fn imbalance(&self) -> f64 {
        let bid = self.total_bid_depth();
        let ask = self.total_ask_depth();
        let total = bid + ask;
        if total > 0.0 {
            (bid - ask) / total
        } else {
            0.0
        }
    }

    /// Estimate price impact of a buy order of given size.
    /// Walks up the ask side of the book.
    /// Returns +inf if the requested size cannot be fully filled.
    pub fn buy_impact(&self, size: f64) -> f64 {
        let mut remaining = size;
        let mut cost = 0.0;

        for level in &self.asks {
            if remaining <= 0.0 {
                break;
            }
            let fill = remaining.min(level.size);
            cost += fill * level.price;
            remaining -= fill;
        }

        if size <= 0.0 {
            self.best_ask()
        } else if remaining > 0.0 {
            f64::INFINITY
        } else {
            cost / size
        }
    }

    /// Estimate price impact of a sell order of given size.
    /// Walks down the bid side of the book.
    pub fn sell_impact(&self, size: f64) -> f64 {
        let mut remaining = size;
        let mut proceeds = 0.0;

        for level in &self.bids {
            if remaining <= 0.0 {
                break;
            }
            let fill = remaining.min(level.size);
            proceeds += fill * level.price;
            remaining -= fill;
        }

        if size > 0.0 {
            proceeds / size
        } else {
            self.best_bid()
        }
    }

    /// Kyle's lambda: estimated price impact per dollar of volume.
    /// Uses a simple linear regression between fill size and execution price.
    pub fn kyle_lambda(&self) -> f64 {
        let mid = self.mid_price();
        let test_size = 100.0; // test with $100

        let buy_price = self.buy_impact(test_size);
        let sell_price = self.sell_impact(test_size);

        let buy_impact = buy_price - mid;
        let sell_impact = mid - sell_price;

        // Average absolute impact per dollar
        (buy_impact.abs() + sell_impact.abs()) / (2.0 * test_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::connectors::polymarket::{OrderBookSnapshot, PriceLevel};

    fn make_snapshot() -> OrderBookSnapshot {
        OrderBookSnapshot {
            market: "0xtest".to_string(),
            asset_id: "12345".to_string(),
            timestamp: "1771321229354".to_string(),
            hash: "abc123".to_string(),
            bids: vec![
                PriceLevel {
                    price: "0.50".to_string(),
                    size: "100".to_string(),
                },
                PriceLevel {
                    price: "0.48".to_string(),
                    size: "200".to_string(),
                },
                PriceLevel {
                    price: "0.45".to_string(),
                    size: "500".to_string(),
                },
            ],
            asks: vec![
                PriceLevel {
                    price: "0.52".to_string(),
                    size: "150".to_string(),
                },
                PriceLevel {
                    price: "0.55".to_string(),
                    size: "300".to_string(),
                },
                PriceLevel {
                    price: "0.60".to_string(),
                    size: "1000".to_string(),
                },
            ],
            min_order_size: "5".to_string(),
            tick_size: "0.01".to_string(),
            neg_risk: false,
            last_trade_price: "0.51".to_string(),
        }
    }

    #[test]
    fn test_mid_price() {
        let mut book = OrderBook::new("test".to_string());
        book.update_from_snapshot(&make_snapshot());
        assert!((book.mid_price() - 0.51).abs() < 0.001);
    }

    #[test]
    fn test_spread() {
        let mut book = OrderBook::new("test".to_string());
        book.update_from_snapshot(&make_snapshot());
        assert!((book.spread() - 0.02).abs() < 0.001);
    }

    #[test]
    fn test_imbalance() {
        let mut book = OrderBook::new("test".to_string());
        book.update_from_snapshot(&make_snapshot());
        let bid_depth = 100.0 + 200.0 + 500.0; // 800
        let ask_depth = 150.0 + 300.0 + 1000.0; // 1450
        let expected = (bid_depth - ask_depth) / (bid_depth + ask_depth);
        assert!((book.imbalance() - expected).abs() < 0.001);
    }

    #[test]
    fn test_buy_impact() {
        let mut book = OrderBook::new("test".to_string());
        book.update_from_snapshot(&make_snapshot());
        // Buy 100 units: fills 100 at ask[0] price 0.52
        let impact = book.buy_impact(100.0);
        assert!((impact - 0.52).abs() < 0.001);
    }

    #[test]
    fn test_buy_impact_crosses_levels() {
        let mut book = OrderBook::new("test".to_string());
        book.update_from_snapshot(&make_snapshot());
        // Buy 200 units: 150 at 0.52 + 50 at 0.55
        let impact = book.buy_impact(200.0);
        let expected = (150.0 * 0.52 + 50.0 * 0.55) / 200.0;
        assert!((impact - expected).abs() < 0.001);
    }

    #[test]
    fn test_buy_impact_insufficient_depth() {
        let mut book = OrderBook::new("test".to_string());
        book.update_from_snapshot(&make_snapshot());
        // Ask depth is 1450, so this cannot be fully filled.
        let impact = book.buy_impact(2000.0);
        assert!(impact.is_infinite());
    }

    #[test]
    fn test_kyle_lambda_positive() {
        let mut book = OrderBook::new("test".to_string());
        book.update_from_snapshot(&make_snapshot());
        assert!(book.kyle_lambda() > 0.0);
    }
}
