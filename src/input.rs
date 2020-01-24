/// Types that can be used as input.
pub trait Input {
    /// The float codification of this type.
    fn as_float(&self) -> f64;
}

impl Input for f64 {
    fn as_float(&self) -> f64 {
        *self
    }
}
