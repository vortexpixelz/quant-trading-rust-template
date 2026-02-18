//! Polymarket CLOB API connector.
//!
//! Implements the public endpoints discovered during API exploration:
//! - /book — live orderbook snapshot
//! - /sampling-markets — active orderbook-enabled markets
//! - /orderbook-history — historical snapshots (time-windowed pagination)
//! - /last-trade-price — most recent trade
//! - /spread — current bid-ask spread
//!
//! Key constraints codified from live testing:
//! - startTs REQUIRED for /orderbook-history (400 without)
//! - limit hard cap 500, use time-windowed pagination
//! - /prices-history broken for windows > minutes
//! - /trades requires API key auth

use std::time::{SystemTime, UNIX_EPOCH};

use reqwest::Client;
use serde::{Deserialize, Serialize};
use thiserror::Error;

const BASE_URL: &str = "https://clob.polymarket.com";

#[derive(Error, Debug)]
pub enum PolymarketError {
    #[error("HTTP request failed: {0}")]
    Http(#[from] reqwest::Error),
    #[error("API returned error status {status}: {body}")]
    Api { status: u16, body: String },
    #[error("Configuration error: {0}")]
    Config(String),
}

// ── Response types ──────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBookSnapshot {
    pub market: String,
    pub asset_id: String,
    pub timestamp: String,
    pub hash: String,
    pub bids: Vec<PriceLevel>,
    pub asks: Vec<PriceLevel>,
    pub min_order_size: String,
    pub tick_size: String,
    pub neg_risk: bool,
    #[serde(default)]
    pub last_trade_price: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriceLevel {
    pub price: String,
    pub size: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBookHistoryResponse {
    pub count: u64,
    pub data: Vec<OrderBookSnapshot>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Token {
    pub token_id: String,
    pub outcome: String,
    #[serde(default)]
    pub price: f64,
    #[serde(default)]
    pub winner: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Market {
    pub condition_id: String,
    pub question: String,
    #[serde(default)]
    pub market_slug: String,
    #[serde(default)]
    pub end_date_iso: String,
    #[serde(default)]
    pub neg_risk: bool,
    #[serde(default)]
    pub minimum_order_size: f64,
    #[serde(default)]
    pub minimum_tick_size: f64,
    #[serde(default)]
    pub tokens: Vec<Token>,
    #[serde(default)]
    pub enable_order_book: bool,
    #[serde(default)]
    pub active: bool,
}

// ── Client ──────────────────────────────────────────────────────────────────

pub struct PolymarketClient {
    client: Client,
    base_url: String,
}

impl PolymarketClient {
    pub fn new(base_url: Option<&str>) -> Result<Self, PolymarketError> {
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .user_agent("quant-trading-rust/0.1")
            .gzip(true)
            .build()?;

        Ok(Self {
            client,
            base_url: base_url.unwrap_or(BASE_URL).to_string(),
        })
    }

    /// Fetch current live orderbook for a token.
    pub async fn get_book(&self, token_id: &str) -> Result<OrderBookSnapshot, PolymarketError> {
        let url = format!("{}/book", self.base_url);
        let resp = self
            .client
            .get(&url)
            .query(&[("token_id", token_id)])
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status().as_u16();
            let body = resp.text().await.unwrap_or_default();
            return Err(PolymarketError::Api { status, body });
        }

        Ok(resp.json().await?)
    }

    /// Fetch active orderbook-enabled markets.
    pub async fn get_sampling_markets(&self) -> Result<Vec<Market>, PolymarketError> {
        let url = format!("{}/sampling-markets", self.base_url);
        let resp = self.client.get(&url).send().await?;

        if !resp.status().is_success() {
            let status = resp.status().as_u16();
            let body = resp.text().await.unwrap_or_default();
            return Err(PolymarketError::Api { status, body });
        }

        Ok(resp.json().await?)
    }

    /// Fetch a page of orderbook history with time-windowed pagination.
    ///
    /// CRITICAL: startTs is required or the API returns 400.
    /// limit is capped at 500 by the server.
    pub async fn get_orderbook_history(
        &self,
        asset_id: &str,
        start_ts_ms: u64,
        end_ts_ms: Option<u64>,
        limit: Option<u32>,
    ) -> Result<OrderBookHistoryResponse, PolymarketError> {
        let url = format!("{}/orderbook-history", self.base_url);
        let mut params = vec![
            ("asset_id".to_string(), asset_id.to_string()),
            ("startTs".to_string(), start_ts_ms.to_string()),
            ("limit".to_string(), limit.unwrap_or(500).min(500).to_string()),
        ];

        if let Some(end) = end_ts_ms {
            params.push(("endTs".to_string(), end.to_string()));
        }

        let resp = self.client.get(&url).query(&params).send().await?;

        if !resp.status().is_success() {
            let status = resp.status().as_u16();
            let body = resp.text().await.unwrap_or_default();
            return Err(PolymarketError::Api { status, body });
        }

        Ok(resp.json().await?)
    }

    /// Iterate all orderbook history using time-windowed pagination.
    /// Advances startTs past last record's timestamp after each page.
    pub async fn iter_orderbook_history(
        &self,
        asset_id: &str,
        start_ts_ms: u64,
        end_ts_ms: Option<u64>,
    ) -> Result<Vec<OrderBookSnapshot>, PolymarketError> {
        let mut all_records = Vec::new();
        let mut current_start = start_ts_ms;

        loop {
            let page = self
                .get_orderbook_history(asset_id, current_start, end_ts_ms, Some(500))
                .await?;

            if page.data.is_empty() {
                break;
            }

            let last_ts: u64 = page
                .data
                .last()
                .and_then(|r| r.timestamp.parse().ok())
                .unwrap_or(0);

            let page_len = page.data.len();
            all_records.extend(page.data);

            // Advance past last timestamp
            current_start = last_ts + 1;

            if page_len < 500 {
                break;
            }
        }

        Ok(all_records)
    }

    /// Current unix timestamp in milliseconds.
    pub fn now_ms() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64
    }
}
