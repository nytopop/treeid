# treeid
An implementation of rational buckets for lexically ordered collections.

# References
- [Dan Hazel - Using rational numbers to key nested sets](https://arxiv.org/abs/0806.3115)
- [David W. Matula, Peter Kornerup - An order preserving finite binary encoding of the rationals](https://www.researchgate.net/publication/261204300_An_order_preserving_finite_binary_encoding_of_the_rationals)

# License(s)
MIT

Internally, this project makes use of a partial rust port of [icza/bitio](https://github.com/icza/bitio) in [src/bitter.rs](src/bitter.rs), which is licensed under Apache 2.0. The full license text can be found at [src/bitter.LICENSE](src/bitter.LICENSE). In some areas the behavior has been changed (notably, we ignore certain bits of written integers where the source library did not), and certain method names have been changed to better align with Rust conventions.
