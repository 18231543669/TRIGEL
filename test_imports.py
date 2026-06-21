"""
导入测试脚本
测试所有模块是否能正确导入
"""

def test_imports():
    """测试所有模块的导入"""
    print("=" * 60)
    print("测试模块导入...")
    print("=" * 60)
    
    # 测试config导入
    print("\n1. 测试 config 模块...")
    try:
        from config import device, update_augmentation_params, get_current_augmentation_params
        print("   ✓ config 模块导入成功")
        print(f"   - device: {device}")
    except Exception as e:
        print(f"   ✗ config 模块导入失败: {e}")
    
    # 测试models导入
    print("\n2. 测试 models 包...")
    try:
        from models import ContrastiveLoss, GCNModel, GCNContrastiveModel, ModelFactory
        print("   ✓ models 包导入成功")
        print(f"   - ContrastiveLoss: {ContrastiveLoss}")
        print(f"   - GCNModel: {GCNModel}")
        print(f"   - GCNContrastiveModel: {GCNContrastiveModel}")
        print(f"   - ModelFactory: {ModelFactory}")
    except Exception as e:
        print(f"   ✗ models 包导入失败: {e}")
    
    # 测试utils导入
    print("\n3. 测试 utils 包...")
    try:
        from utils import (
            set_random_seed, construct_adjacency, load_dataset, stratified_split,
            calculate_metrics, calculate_average_results, print_average_results,
            visualize_training,
            create_result_directory, save_results, load_results,
            merge_configs, validate_config, print_config_summary
        )
        print("   ✓ utils 包导入成功")
        print(f"   - data_utils: set_random_seed, construct_adjacency, load_dataset, stratified_split")
        print(f"   - metrics: calculate_metrics, calculate_average_results, print_average_results")
        print(f"   - visualization: visualize_training")
        print(f"   - io_utils: create_result_directory, save_results, load_results, merge_configs, validate_config, print_config_summary")
    except Exception as e:
        print(f"   ✗ utils 包导入失败: {e}")
    
    # 测试augmentation导入
    print("\n4. 测试 augmentation 包...")
    try:
        from augmentation import (
            PageRankCalculator, MinorityClassIdentifier, 
            FeatureAugmentor, TopologyAugmentor, DataAugmentor,
            apply_minority_aware_augmentation,
            AugmentationEffectsReporter, print_augmentation_effects
        )
        print("   ✓ augmentation 包导入成功")
        print(f"   - augmentors: PageRankCalculator, MinorityClassIdentifier, FeatureAugmentor, TopologyAugmentor, DataAugmentor")
        print(f"   - reporter: AugmentationEffectsReporter, print_augmentation_effects")
    except Exception as e:
        print(f"   ✗ augmentation 包导入失败: {e}")
    
    # 测试trainer导入
    print("\n5. 测试 trainer 模块...")
    try:
        from trainer import train_with_contrastive_learning, evaluate_model
        print("   ✓ trainer 模块导入成功")
        print(f"   - train_with_contrastive_learning: {train_with_contrastive_learning}")
        print(f"   - evaluate_model: {evaluate_model}")
    except Exception as e:
        print(f"   ✗ trainer 模块导入失败: {e}")
    
    print("\n" + "=" * 60)
    print("导入测试完成！")
    print("=" * 60)


if __name__ == "__main__":
    test_imports()
