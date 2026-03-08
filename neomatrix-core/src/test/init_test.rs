//! Test suite for weight initialization strategies
//!
//! Tests verify Random, Xavier, and He initialization produce correct shapes
//! and distributions for neural network weight matrices.

use crate::layers::init::Init;

#[cfg(test)]
mod init_tests {
    use super::*;

    #[test]
    fn random_init_default_range() {
        let weights = Init::Random.init(10, 5, None);

        assert_eq!(weights.shape, vec![10, 5]);
        assert_eq!(weights.dimension, 2);
        assert_eq!(weights.data.len(), 50);

        assert!(weights.data.iter().all(|&x| x >= -1.0 && x < 1.0));
    }

    #[test]
    fn random_init_custom_range() {
        let weights = Init::Random.init(10, 5, Some(0.0..10.0));

        assert_eq!(weights.shape, vec![10, 5]);
        assert_eq!(weights.dimension, 2);

        assert!(weights.data.iter().all(|&x| x >= 0.0 && x < 10.0));
    }

    #[test]
    fn random_init_negative_range() {
        let weights = Init::Random.init(5, 5, Some(-5.0..0.0));

        assert!(weights.data.iter().all(|&x| x >= -5.0 && x < 0.0));
    }

    #[test]
    fn random_init_produces_different_values() {
        let w1 = Init::Random.init(10, 10, None);
        let w2 = Init::Random.init(10, 10, None);

        let different_count = w1
            .data
            .iter()
            .zip(w2.data.iter())
            .filter(|&(&a, &b)| (a - b).abs() > 0.001)
            .count();

        assert!(different_count > 90);
    }

    #[test]
    fn xavier_init_shape() {
        let weights = Init::Xavier.init(100, 50, None);

        assert_eq!(weights.shape, vec![100, 50]);
        assert_eq!(weights.dimension, 2);
        assert_eq!(weights.data.len(), 5000);
    }

    #[test]
    fn xavier_init_distribution() {
        let in_feat = 100;
        let out_feat = 50;
        let weights = Init::Xavier.init(in_feat, out_feat, None);

        let mean: f32 = weights.data.iter().sum::<f32>() / weights.data.len() as f32;
        assert!(mean.abs() < 0.1);

        let expected_std = (2.0 / (in_feat + out_feat) as f32).sqrt();

        let variance: f32 = weights
            .data
            .iter()
            .map(|&x| (x - mean).powf(2.0))
            .sum::<f32>()
            / weights.data.len() as f32;
        let actual_std = variance.sqrt();

        assert!(
            (actual_std - expected_std).abs() < 0.05,
            "Xavier std dev {} should be close to {}",
            actual_std,
            expected_std
        );
    }

    #[test]
    fn xavier_init_symmetric_distribution() {
        let weights = Init::Xavier.init(50, 50, None);

        let positive_count = weights.data.iter().filter(|&&x| x > 0.0).count();
        let negative_count = weights.data.iter().filter(|&&x| x < 0.0).count();

        let ratio = positive_count as f32 / negative_count as f32;
        assert!(
            ratio > 0.8 && ratio < 1.2,
            "Xavier should produce roughly equal positive/negative values"
        );
    }

    #[test]
    fn xavier_init_ignores_range_parameter() {
        let w1 = Init::Xavier.init(10, 10, None);
        let w2 = Init::Xavier.init(10, 10, Some(0.0..100.0));

        let mean1: f32 = w1.data.iter().sum::<f32>() / w1.data.len() as f32;
        let mean2: f32 = w2.data.iter().sum::<f32>() / w2.data.len() as f32;

        assert!(mean1.abs() < 0.2);
        assert!(mean2.abs() < 0.2);
    }

    #[test]
    fn he_init_shape() {
        let weights = Init::He.init(100, 50, None);

        assert_eq!(weights.shape, vec![100, 50]);
        assert_eq!(weights.dimension, 2);
        assert_eq!(weights.data.len(), 5000);
    }

    #[test]
    fn he_init_distribution() {
        let in_feat = 100;
        let out_feat = 50;
        let weights = Init::He.init(in_feat, out_feat, None);

        let mean: f32 = weights.data.iter().sum::<f32>() / weights.data.len() as f32;
        assert!(mean.abs() < 0.1);

        let expected_std = (2.0 / in_feat as f32).sqrt();

        let variance: f32 = weights
            .data
            .iter()
            .map(|&x| (x - mean).powf(2.0))
            .sum::<f32>()
            / weights.data.len() as f32;
        let actual_std = variance.sqrt();

        assert!(
            (actual_std - expected_std).abs() < 0.05,
            "He std dev {} should be close to {}",
            actual_std,
            expected_std
        );
    }

    #[test]
    fn he_init_only_uses_in_features() {
        let w1 = Init::He.init(100, 50, None);
        let w2 = Init::He.init(100, 200, None);

        let mean1: f32 = w1.data.iter().sum::<f32>() / w1.data.len() as f32;
        let mean2: f32 = w2.data.iter().sum::<f32>() / w2.data.len() as f32;

        let var1: f32 =
            w1.data.iter().map(|&x| (x - mean1).powf(2.0)).sum::<f32>() / w1.data.len() as f32;
        let var2: f32 =
            w2.data.iter().map(|&x| (x - mean2).powf(2.0)).sum::<f32>() / w2.data.len() as f32;

        assert!(
            (var1 - var2).abs() < 0.01,
            "He init variance should be similar when in_feat is the same"
        );
    }

    #[test]
    fn he_init_symmetric_distribution() {
        let weights = Init::He.init(50, 50, None);

        let positive_count = weights.data.iter().filter(|&&x| x > 0.0).count();
        let negative_count = weights.data.iter().filter(|&&x| x < 0.0).count();

        let ratio = positive_count as f32 / negative_count as f32;
        assert!(
            ratio > 0.8 && ratio < 1.2,
            "He should produce roughly equal positive/negative values"
        );
    }

    #[test]
    fn he_init_ignores_range_parameter() {
        let w1 = Init::He.init(10, 10, None);
        let w2 = Init::He.init(10, 10, Some(0.0..100.0));

        let mean1: f32 = w1.data.iter().sum::<f32>() / w1.data.len() as f32;
        let mean2: f32 = w2.data.iter().sum::<f32>() / w2.data.len() as f32;

        assert!(mean1.abs() < 0.3);
        assert!(mean2.abs() < 0.3);
    }

    #[test]
    fn all_init_methods_produce_2d_tensors() {
        let random = Init::Random.init(5, 3, None);
        let xavier = Init::Xavier.init(5, 3, None);
        let he = Init::He.init(5, 3, None);

        assert_eq!(random.dimension, 2);
        assert_eq!(xavier.dimension, 2);
        assert_eq!(he.dimension, 2);
    }

    #[test]
    fn all_init_methods_match_requested_shape() {
        let in_feat = 20;
        let out_feat = 10;

        let random = Init::Random.init(in_feat, out_feat, None);
        let xavier = Init::Xavier.init(in_feat, out_feat, None);
        let he = Init::He.init(in_feat, out_feat, None);

        assert_eq!(random.shape, vec![in_feat, out_feat]);
        assert_eq!(xavier.shape, vec![in_feat, out_feat]);
        assert_eq!(he.shape, vec![in_feat, out_feat]);
    }

    #[test]
    fn init_methods_handle_large_matrices() {
        let in_feat = 1000;
        let out_feat = 500;

        let random = Init::Random.init(in_feat, out_feat, None);
        let xavier = Init::Xavier.init(in_feat, out_feat, None);
        let he = Init::He.init(in_feat, out_feat, None);

        assert_eq!(random.data.len(), in_feat * out_feat);
        assert_eq!(xavier.data.len(), in_feat * out_feat);
        assert_eq!(he.data.len(), in_feat * out_feat);

        assert!(random.data.iter().all(|&x| x.is_finite()));
        assert!(xavier.data.iter().all(|&x| x.is_finite()));
        assert!(he.data.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn init_methods_handle_single_neuron() {
        let random = Init::Random.init(1, 1, None);
        let xavier = Init::Xavier.init(1, 1, None);
        let he = Init::He.init(1, 1, None);

        assert_eq!(random.shape, vec![1, 1]);
        assert_eq!(xavier.shape, vec![1, 1]);
        assert_eq!(he.shape, vec![1, 1]);
    }

    #[test]
    fn xavier_vs_he_variance_difference() {
        let in_feat = 100;
        let out_feat = 50;

        let xavier = Init::Xavier.init(in_feat, out_feat, None);
        let he = Init::He.init(in_feat, out_feat, None);

        let xavier_mean: f32 = xavier.data.iter().sum::<f32>() / xavier.data.len() as f32;
        let he_mean: f32 = he.data.iter().sum::<f32>() / he.data.len() as f32;

        let xavier_var: f32 = xavier
            .data
            .iter()
            .map(|&x| (x - xavier_mean).powf(2.0))
            .sum::<f32>()
            / xavier.data.len() as f32;

        let he_var: f32 = he
            .data
            .iter()
            .map(|&x| (x - he_mean).powf(2.0))
            .sum::<f32>()
            / he.data.len() as f32;

        assert!(
            he_var > xavier_var,
            "He initialization should have larger variance than Xavier for same in_feat"
        );
    }
}
