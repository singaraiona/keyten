//! End-to-end fixture: build `1 2 3` and atom `10`, run `dispatch_plus`,
//! print and assert the result.

use keyten::alloc::{alloc_atom, alloc_vec_i64};
use keyten::ctx::Ctx;
use keyten::kind::Kind;
use keyten::obj::attr_flags;
use keyten::op::dispatch_plus;

fn main() {
    let ctx = Ctx::quiet();
    unsafe {
        let mut x = alloc_vec_i64(&ctx, 3);
        x.as_mut_slice::<i64>().copy_from_slice(&[1, 2, 3]);
        let y = alloc_atom(Kind::I64, 10i64);

        let r = dispatch_plus(x, y, &ctx).expect("plus failed");

        assert_eq!(r.kind(), Kind::I64);
        assert_eq!(r.as_slice::<i64>(), &[11, 12, 13]);
        assert_eq!(r.attr() & attr_flags::HAS_NULLS, 0);

        let formatted = r
            .as_slice::<i64>()
            .iter()
            .map(|n| n.to_string())
            .collect::<Vec<_>>()
            .join(" ");
        println!("1 2 3 + 10 -> {formatted}");
    }
}
