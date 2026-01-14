use crate::codec::error::{CodecError, CodecResult};
use core::ops::{Add, BitXor, Div, Mul};

/// Reduce 抽象：把多路输出合成为一路输出。
///
/// 约定：
/// - `items.len() == branches`（由 ForkJoinNode 保证）
/// - reduce 失败返回 `CodecError`
pub trait Reduce<T>: Send + Sync + 'static {
    fn reduce(&self, items: &[T]) -> CodecResult<T>;
}

impl<T, F> Reduce<T> for F
where
    F: Fn(&[T]) -> CodecResult<T> + Send + Sync + 'static,
{
    fn reduce(&self, items: &[T]) -> CodecResult<T> {
        (self)(items)
    }
}

/// 预置 Reduce：加权求和
///
/// - `weight == None`：视为全 1
/// - `weight == Some(vec)`：长度必须与分支数一致
///
/// 计算：`out[0]*weight[0] + out[1]*weight[1] + ...`
#[derive(Clone, Debug)]
pub struct Sum<T> {
    pub weight: Option<Vec<T>>,
}

impl<T> Sum<T> {
    pub fn new(weight: Option<Vec<T>>) -> Self {
        Self { weight }
    }
}

impl<T> Default for Sum<T> {
    fn default() -> Self {
        Self { weight: None }
    }
}

impl<T> Reduce<T> for Sum<T>
where
    T: Copy + Default + From<u8> + Add<Output = T> + Mul<Output = T> + Send + Sync + 'static,
{
    fn reduce(&self, items: &[T]) -> CodecResult<T> {
        if items.is_empty() {
            return Err(CodecError::InvalidData("sum reduce expects non-empty items"));
        }

        let w = if let Some(w) = &self.weight {
            if w.len() != items.len() {
                return Err(CodecError::InvalidData("sum reduce weight length mismatch"));
            }
            Some(w.as_slice())
        } else {
            None
        };

        let one: T = 1u8.into();
        let mut acc: T = T::default();
        for (i, &v) in items.iter().enumerate() {
            let wi = w.map(|ws| ws[i]).unwrap_or(one);
            acc = acc + (v * wi);
        }
        Ok(acc)
    }
}

/// 预置 Reduce：加权求积
///
/// - `weight == None`：视为全 1
/// - `weight == Some(vec)`：长度必须与分支数一致
///
/// 计算：`(out[0]*weight[0]) * (out[1]*weight[1]) * ...`
#[derive(Clone, Debug)]
pub struct Product<T> {
    pub weight: Option<Vec<T>>,
}

impl<T> Product<T> {
    pub fn new(weight: Option<Vec<T>>) -> Self {
        Self { weight }
    }
}

impl<T> Default for Product<T> {
    fn default() -> Self {
        Self { weight: None }
    }
}

impl<T> Reduce<T> for Product<T>
where
    T: Copy + From<u8> + Mul<Output = T> + Send + Sync + 'static,
{
    fn reduce(&self, items: &[T]) -> CodecResult<T> {
        if items.is_empty() {
            return Err(CodecError::InvalidData("product reduce expects non-empty items"));
        }

        let w = if let Some(w) = &self.weight {
            if w.len() != items.len() {
                return Err(CodecError::InvalidData("product reduce weight length mismatch"));
            }
            Some(w.as_slice())
        } else {
            None
        };

        let one: T = 1u8.into();
        let mut acc: T = one;
        for (i, &v) in items.iter().enumerate() {
            let wi = w.map(|ws| ws[i]).unwrap_or(one);
            acc = acc * (v * wi);
        }
        Ok(acc)
    }
}

/// 预置 Reduce：求平均（Mean）
///
/// - 不支持 weight
/// - 计算：`(out[0] + out[1] + ...)/N`
///
/// 注意：
/// - 对整数类型会发生“截断除法”（例如 10/4 = 2）
/// - 对浮点类型得到常规平均值
#[derive(Clone, Copy, Debug, Default)]
pub struct Mean;

impl<T> Reduce<T> for Mean
where
    T: Copy + Default + From<u8> + Add<Output = T> + Div<Output = T> + Send + Sync + 'static,
{
    fn reduce(&self, items: &[T]) -> CodecResult<T> {
        if items.is_empty() {
            return Err(CodecError::InvalidData("mean reduce expects non-empty items"));
        }

        let one: T = 1u8.into();
        let mut sum: T = T::default();
        for &v in items {
            sum = sum + v;
        }

        // 用 repeated-add 构造 N（避免额外依赖/类型转换）
        let mut n: T = T::default();
        for _ in 0..items.len() {
            n = n + one;
        }
        Ok(sum / n)
    }
}

/// 预置 Reduce：最大值（Max）
#[derive(Clone, Copy, Debug, Default)]
pub struct Max;

impl<T> Reduce<T> for Max
where
    T: Ord + Clone + Send + Sync + 'static,
{
    fn reduce(&self, items: &[T]) -> CodecResult<T> {
        let Some(first) = items.first() else {
            return Err(CodecError::InvalidData("max reduce expects non-empty items"));
        };
        let mut best = first.clone();
        for v in items.iter().skip(1) {
            if *v > best {
                best = v.clone();
            }
        }
        Ok(best)
    }
}

/// 预置 Reduce：最小值（Min）
#[derive(Clone, Copy, Debug, Default)]
pub struct Min;

impl<T> Reduce<T> for Min
where
    T: Ord + Clone + Send + Sync + 'static,
{
    fn reduce(&self, items: &[T]) -> CodecResult<T> {
        let Some(first) = items.first() else {
            return Err(CodecError::InvalidData("min reduce expects non-empty items"));
        };
        let mut best = first.clone();
        for v in items.iter().skip(1) {
            if *v < best {
                best = v.clone();
            }
        }
        Ok(best)
    }
}

/// 预置 Reduce：拼接（Concat）
#[derive(Clone, Copy, Debug, Default)]
pub struct Concat;

impl<T> Reduce<Vec<T>> for Concat
where
    T: Clone + Send + Sync + 'static,
{
    fn reduce(&self, items: &[Vec<T>]) -> CodecResult<Vec<T>> {
        if items.is_empty() {
            return Err(CodecError::InvalidData("concat reduce expects non-empty items"));
        }
        let total: usize = items.iter().map(|v| v.len()).sum();
        let mut out = Vec::with_capacity(total);
        for v in items {
            out.extend_from_slice(v);
        }
        Ok(out)
    }
}

impl Reduce<String> for Concat {
    fn reduce(&self, items: &[String]) -> CodecResult<String> {
        if items.is_empty() {
            return Err(CodecError::InvalidData("concat reduce expects non-empty items"));
        }
        let total: usize = items.iter().map(|s| s.len()).sum();
        let mut out = String::with_capacity(total);
        for s in items {
            out.push_str(s);
        }
        Ok(out)
    }
}

/// 预置 Reduce：异或（Xor）
///
/// 计算：`out[0] ^ out[1] ^ ...`，单位元为 0（`Default`）
#[derive(Clone, Copy, Debug, Default)]
pub struct Xor;

impl<T> Reduce<T> for Xor
where
    T: Copy + Default + BitXor<Output = T> + Send + Sync + 'static,
{
    fn reduce(&self, items: &[T]) -> CodecResult<T> {
        if items.is_empty() {
            return Err(CodecError::InvalidData("xor reduce expects non-empty items"));
        }
        let mut acc: T = T::default();
        for &v in items {
            acc = acc ^ v;
        }
        Ok(acc)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sum_default_weight() {
        let r = Sum::<i32>::default();
        assert_eq!(r.reduce(&[1, 2, 3]).unwrap(), 6);
    }

    #[test]
    fn sum_with_weight() {
        let r = Sum::new(Some(vec![2i32, 3, 4]));
        assert_eq!(r.reduce(&[1, 2, 3]).unwrap(), 1 * 2 + 2 * 3 + 3 * 4);
    }

    #[test]
    fn sum_weight_len_mismatch() {
        let r = Sum::new(Some(vec![1i32]));
        assert!(matches!(r.reduce(&[1, 2]), Err(CodecError::InvalidData(_))));
    }

    #[test]
    fn product_default_weight() {
        let r = Product::<i32>::default();
        assert_eq!(r.reduce(&[2, 3, 4]).unwrap(), 24);
    }

    #[test]
    fn product_with_weight() {
        let r = Product::new(Some(vec![2i32, 3, 4]));
        assert_eq!(r.reduce(&[1, 2, 3]).unwrap(), (1 * 2) * (2 * 3) * (3 * 4));
    }

    #[test]
    fn product_weight_len_mismatch() {
        let r = Product::new(Some(vec![1i32]));
        assert!(matches!(r.reduce(&[1, 2]), Err(CodecError::InvalidData(_))));
    }

    #[test]
    fn mean_i32_trunc_div() {
        let r = Mean;
        // (1+2+3+4)/4 = 2 (整数除法截断)
        assert_eq!(r.reduce(&[1i32, 2, 3, 4]).unwrap(), 2);
    }

    #[test]
    fn mean_f64() {
        let r = Mean;
        let v = r.reduce(&[1.0f64, 2.0, 3.0]).unwrap();
        assert!((v - 2.0).abs() < 1e-12);
    }

    #[test]
    fn max_i32() {
        let r = Max;
        assert_eq!(r.reduce(&[1i32, 9, 3]).unwrap(), 9);
    }

    #[test]
    fn min_i32() {
        let r = Min;
        assert_eq!(r.reduce(&[1i32, -9, 3]).unwrap(), -9);
    }

    #[test]
    fn concat_vec() {
        let r = Concat;
        let out = r.reduce(&[vec![1, 2], vec![3], vec![]]).unwrap();
        assert_eq!(out, vec![1, 2, 3]);
    }

    #[test]
    fn concat_string() {
        let r = Concat;
        let out = r
            .reduce(&["ab".to_string(), "-".to_string(), "cd".to_string()])
            .unwrap();
        assert_eq!(out, "ab-cd");
    }

    #[test]
    fn xor_u8() {
        let r = Xor;
        // 0b1100 ^ 0b1010 = 0b0110
        assert_eq!(r.reduce(&[0b1100u8, 0b1010u8]).unwrap(), 0b0110u8);
    }
}

