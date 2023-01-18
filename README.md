# Avail Light Client Sampling Simulator

Run `cargo run --example <simulation> --release` to generate a csv of data. Per example in `./examples` there should also be a jupyter notebook with plots. Release builds are substantially faster than debug builds, so make sure to run with `--release`.

Setting up a new simulation is easy! Just make a rust file in `./examples`, add it to the `Cargo.toml` with 
```toml
[[example]]
name = "my_example"
```
and set up an experiment (or a series of experiments) in your example file. 

Experiments can be set up by making an `ExperimentConfig` like so
```rust
let e = ExperimentConfig {
    n, // The width/height of the non-erasure encoded matrix. Currently assumed to be square.
    dims, // The number of dimenions to do erasure encoding in. Either 1 or 2.
    n_clients, // The number of light clients present
    percent_censored, // The percentage of light clients being censored
    n_samples, // The number of samples each light client performs
    sample_strategy: SampleStrategy::RandomPoints, // The SampleStrategy
};
```

Available sampling strategies are
```rust
pub enum SampleStrategy {
    /// Split the grid into width x height chunks, then sample those
    Box {
        /// How wide each chunk is. Must evenly divide the grid width.
        width: usize,
        /// same as `width` but for columns
        height: usize,
    },
    /// Sample cells uniformly at random
    RandomPoints,
}
```

To run an experiment, just call `ExperimentConfig::run`. 
Use the present examples as a reference for running experiments in parallel, saving outputs to a csv, etc.

