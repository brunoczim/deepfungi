macro_rules! assert_float_eq {
    ($left:expr, $right:expr, $precision:expr $(,)?) => {{
        let left = $left;
        let right = $right;
        let precision = $precision;
        if (left - right).abs() > precision {
            panic!(
                "assertion failed: `(left == right)`\n left: `{:?}`,\n right: \
                 `{:?}`\n precision: {:?}",
                left, right, precision,
            )
        }
    }};
    ($left:expr, $right:expr, $precision:expr, $($arg:tt)+) => {{
        let left = $left;
        let right = $right;
        let precision = $precision;
        if (left - right).abs() > precision {
            panic!(
                "assertion failed: `(left == right)`\n left: `{:?}`,\n right: \
                 `{:?}`\n precision: {:?}\n message: {}",
                left, right, precision, format_args!($($arg)*)
            )
        }
    }};
}
