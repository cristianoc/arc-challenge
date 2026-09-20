//! Stable SymArc core, shared by the solver CLI and isolated experiments.
//! Experimental policies belong in experiments/, not in this library.
pub mod dsl;
pub mod grid;
mod json;
pub mod random;
pub mod report;
pub mod search;
pub mod task;
pub use task::Task;

#[cfg(test)]
mod tests;
