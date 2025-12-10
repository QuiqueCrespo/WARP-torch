"""
Test file for PyTorch models with dummy data.
Tests all model variants: WSM, GRU, LSTM, FFNN
"""

import torch
import torch.nn as nn
import numpy as np
from models_torch import (
    RootMLP_Regressor,
    RootMLP_Classif,
    GradualMLP,
    ConvNet1D,
    WSM,
    GRU,
    LSTM,
    FFNN,
    make_model
)
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def print_test_header(test_name):
    """Print a formatted test header."""
    print("\n" + "="*70)
    print(f"  {test_name}")
    print("="*70)


def test_root_mlp_regressor():
    """Test RootMLP_Regressor with and without uncertainty."""
    print_test_header("Test 1: RootMLP_Regressor")

    batch_size, input_dim, output_dim = 8, 5, 3

    # Test with uncertainty prediction
    model = RootMLP_Regressor(
        input_dim=input_dim,
        output_dim=output_dim,
        width_size=64,
        depth=3,
        predict_uncertainty=True
    )

    x = torch.randn(batch_size, input_dim)
    output = model(x)

    expected_shape = (batch_size, output_dim * 2)  # mean + std
    assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
    print(f"✓ With uncertainty: Input {x.shape} -> Output {output.shape}")

    # Test without uncertainty prediction
    model_no_unc = RootMLP_Regressor(
        input_dim=input_dim,
        output_dim=output_dim,
        width_size=64,
        depth=3,
        predict_uncertainty=False
    )

    output_no_unc = model_no_unc(x)
    expected_shape_no_unc = (batch_size, output_dim)
    assert output_no_unc.shape == expected_shape_no_unc, f"Expected {expected_shape_no_unc}, got {output_no_unc.shape}"
    print(f"✓ Without uncertainty: Input {x.shape} -> Output {output_no_unc.shape}")

    # Test gradient flow
    loss = output.sum()
    loss.backward()
    assert model.network[0].weight.grad is not None, "Gradients not flowing"
    print(f"✓ Gradients flowing correctly")


def test_root_mlp_classif():
    """Test RootMLP_Classif for classification."""
    print_test_header("Test 2: RootMLP_Classif")

    batch_size, nb_classes = 16, 10

    model = RootMLP_Classif(
        nb_classes=nb_classes,
        width_size=128,
        depth=2,
        positional_enc_dim=4
    )

    x = torch.randn(batch_size, 1 + 4)  # 1 + positional_enc_dim
    output = model(x)

    expected_shape = (batch_size, nb_classes)
    assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
    print(f"✓ Classification: Input {x.shape} -> Output {output.shape}")

    # Test with softmax
    probs = torch.softmax(output, dim=-1)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(batch_size)), "Probabilities don't sum to 1"
    print(f"✓ Softmax probabilities sum to 1")


def test_gradual_mlp():
    """Test GradualMLP with different hidden layer counts."""
    print_test_header("Test 3: GradualMLP")

    batch_size, input_dim, output_dim = 4, 100, 10
    x = torch.randn(batch_size, input_dim)

    for hidden_layers in [0, 1, 2]:
        model = GradualMLP(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_layers=hidden_layers,
            activation=nn.Tanh()
        )

        output = model(x)
        expected_shape = (batch_size, output_dim)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        print(f"✓ hidden_layers={hidden_layers}: Input {x.shape} -> Output {output.shape}")


def test_conv_net_1d():
    """Test ConvNet1D for 1D data."""
    print_test_header("Test 4: ConvNet1D")

    batch_size, in_channels, time_steps = 8, 3, 50
    out_channels, hidden_channels = 5, 32

    model = ConvNet1D(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=hidden_channels,
        kernel_size=5,
        n_layers=3
    )

    x = torch.randn(batch_size, in_channels, time_steps)
    output = model(x)

    expected_shape = (batch_size, out_channels, time_steps)
    assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
    print(f"✓ Conv1D: Input {x.shape} -> Output {output.shape}")


def test_wsm_non_autoregressive():
    """Test WSM in non-autoregressive mode."""
    print_test_header("Test 5: WSM (Non-Autoregressive)")

    batch_size, seq_len, data_size = 4, 20, 2

    model = WSM(
        data_size=data_size,
        width_size=32,
        depth=2,
        activation="relu",
        final_activation="tanh",
        predict_uncertainty=True,
        time_as_channel=True,
        forcing_prob=1.0,
        autoregressive_train=False,
        stochastic_ar=False,
        input_prev_data=False
    )

    xs = torch.randn(batch_size, seq_len, data_size)
    ts = torch.linspace(0, 1, seq_len).unsqueeze(0).unsqueeze(-1).expand(batch_size, seq_len, 1)

    output = model(xs, ts)

    expected_shape = (batch_size, seq_len, data_size * 2)  # mean + std
    assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
    print(f"✓ Non-AR: Input {xs.shape} -> Output {output.shape}")

    # Test gradient flow
    loss = output.mean()
    loss.backward()
    assert model.As[0].grad is not None, "Gradients not flowing to A matrix"
    print(f"✓ Gradients flowing correctly")


def test_wsm_autoregressive():
    """Test WSM in autoregressive mode."""
    print_test_header("Test 6: WSM (Autoregressive)")

    batch_size, seq_len, data_size = 4, 15, 2

    model = WSM(
        data_size=data_size,
        width_size=32,
        depth=2,
        activation="relu",
        final_activation="tanh",
        predict_uncertainty=True,
        time_as_channel=True,
        forcing_prob=0.5,
        autoregressive_train=True,
        stochastic_ar=False,
        input_prev_data=False
    )

    xs = torch.randn(batch_size, seq_len, data_size)
    ts = torch.linspace(0, 1, seq_len).unsqueeze(0).unsqueeze(-1).expand(batch_size, seq_len, 1)

    output = model(xs, ts)

    expected_shape = (batch_size, seq_len, data_size * 2)
    assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
    print(f"✓ AR (deterministic): Input {xs.shape} -> Output {output.shape}")

    # Test with inference_start
    output_inference = model(xs, ts, inference_start=0.5)
    assert output_inference.shape == expected_shape, f"Expected {expected_shape}, got {output_inference.shape}"
    print(f"✓ AR with inference_start=0.5: Output {output_inference.shape}")


def test_wsm_stochastic_ar():
    """Test WSM in stochastic autoregressive mode."""
    print_test_header("Test 7: WSM (Stochastic Autoregressive)")

    batch_size, seq_len, data_size = 4, 15, 2

    model = WSM(
        data_size=data_size,
        width_size=32,
        depth=2,
        activation="relu",
        final_activation="tanh",
        predict_uncertainty=True,
        time_as_channel=True,
        forcing_prob=0.5,
        autoregressive_train=True,
        stochastic_ar=True,
        smooth_inference=False,
        input_prev_data=False
    )

    xs = torch.randn(batch_size, seq_len, data_size)
    ts = torch.linspace(0, 1, seq_len).unsqueeze(0).unsqueeze(-1).expand(batch_size, seq_len, 1)

    output = model(xs, ts)

    expected_shape = (batch_size, seq_len, data_size * 2)
    assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
    print(f"✓ Stochastic AR: Input {xs.shape} -> Output {output.shape}")


def test_wsm_with_conv_embedding():
    """Test WSM with convolutional embedding."""
    print_test_header("Test 8: WSM with Conv Embedding")

    batch_size, seq_len, data_size = 4, 20, 3

    model = WSM(
        data_size=data_size,
        width_size=32,
        depth=2,
        activation="relu",
        final_activation="tanh",
        predict_uncertainty=True,
        time_as_channel=True,
        forcing_prob=1.0,
        autoregressive_train=False,
        stochastic_ar=False,
        conv_embedding=(8, 5)  # (out_channels, kernel_size)
    )

    xs = torch.randn(batch_size, seq_len, data_size)
    ts = torch.linspace(0, 1, seq_len).unsqueeze(0).unsqueeze(-1).expand(batch_size, seq_len, 1)

    output = model(xs, ts)

    expected_shape = (batch_size, seq_len, data_size * 2)
    assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
    print(f"✓ With conv embedding: Input {xs.shape} -> Output {output.shape}")


def test_wsm_classification():
    """Test WSM for classification tasks."""
    print_test_header("Test 9: WSM Classification")

    batch_size, seq_len, data_size = 8, 15, 2
    nb_classes = 10

    model = WSM(
        data_size=data_size,
        width_size=64,
        depth=2,
        activation="relu",
        nb_classes=nb_classes,
        time_as_channel=True,
        forcing_prob=1.0,
        autoregressive_train=False,
        stochastic_ar=False,
        positional_encoding=(4,)  # 4-dim positional encoding
    )

    xs = torch.randn(batch_size, seq_len, data_size)
    ts = torch.linspace(0, 1, seq_len).unsqueeze(0).unsqueeze(-1).expand(batch_size, seq_len, 1)

    # Add positional encoding to ts
    pos_enc = torch.randn(batch_size, seq_len, 4)
    ts_with_pos = torch.cat([ts, pos_enc], dim=-1)

    output = model(xs, ts_with_pos)

    expected_shape = (batch_size, seq_len, nb_classes)
    assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
    print(f"✓ Classification: Input {xs.shape} -> Output {output.shape}")


def test_gru_model():
    """Test GRU model."""
    print_test_header("Test 10: GRU Model")

    batch_size, seq_len, data_size = 8, 25, 3
    hidden_size = 64

    model = GRU(
        data_size=data_size,
        hidden_size=hidden_size,
        predict_uncertainty=True,
        time_as_channel=True,
        forcing_prob=0.5
    )

    xs = torch.randn(batch_size, seq_len, data_size)
    ts = torch.linspace(0, 1, seq_len).unsqueeze(0).unsqueeze(-1).expand(batch_size, seq_len, 1)

    output = model(xs, ts)

    expected_shape = (batch_size, seq_len, data_size * 2)
    assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
    print(f"✓ GRU: Input {xs.shape} -> Output {output.shape}")

    # Test with inference_start
    output_inference = model(xs, ts, inference_start=0.5)
    assert output_inference.shape == expected_shape, f"Expected {expected_shape}, got {output_inference.shape}"
    print(f"✓ GRU with inference_start: Output {output_inference.shape}")

    # Test gradient flow
    loss = output.mean()
    loss.backward()
    assert model.cell.weight_ih.grad is not None, "Gradients not flowing"
    print(f"✓ Gradients flowing correctly")


def test_lstm_model():
    """Test LSTM model."""
    print_test_header("Test 11: LSTM Model")

    batch_size, seq_len, data_size = 8, 25, 3
    hidden_size = 64

    model = LSTM(
        data_size=data_size,
        hidden_size=hidden_size,
        predict_uncertainty=True,
        time_as_channel=True,
        forcing_prob=0.5
    )

    xs = torch.randn(batch_size, seq_len, data_size)
    ts = torch.linspace(0, 1, seq_len).unsqueeze(0).unsqueeze(-1).expand(batch_size, seq_len, 1)

    output = model(xs, ts)

    expected_shape = (batch_size, seq_len, data_size * 2)
    assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
    print(f"✓ LSTM: Input {xs.shape} -> Output {output.shape}")

    # Test gradient flow
    loss = output.mean()
    loss.backward()
    assert model.cell.weight_ih.grad is not None, "Gradients not flowing"
    print(f"✓ Gradients flowing correctly")


def test_ffnn_model():
    """Test FFNN model."""
    print_test_header("Test 12: FFNN Model")

    batch_size, seq_len, data_size = 8, 699, 1  # FFNN expects specific sequence length

    model = FFNN(data_size=data_size)

    xs = torch.randn(batch_size, seq_len, data_size)
    ts = torch.linspace(0, 1, seq_len).unsqueeze(0).unsqueeze(-1).expand(batch_size, seq_len, 1)

    output = model(xs, ts)

    expected_shape = (batch_size, seq_len, data_size)
    assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
    print(f"✓ FFNN: Input {xs.shape} -> Output {output.shape}")


def test_make_model_factory():
    """Test make_model factory function."""
    print_test_header("Test 13: make_model Factory")

    data_size, nb_classes = 3, None

    # Test WSM
    config_wsm = {
        'model': {
            'model_type': 'wsm',
            'root_width_size': 32,
            'root_depth': 2,
            'root_activation': 'relu',
            'root_final_activation': 'tanh',
            'init_state_layers': 2,
            'input_prev_data': False,
            'time_as_channel': True,
            'forcing_prob': 1.0,
            'std_lower_bound': 0.01,
            'weights_lim': None,
            'noise_theta_init': None,
            'positional_encoding': None,
            'root_output_dim': None,
            'conv_embedding': None
        },
        'training': {
            'use_nll_loss': True,
            'autoregressive': False,
            'stochastic': False
        }
    }

    model_wsm = make_model(data_size, nb_classes, config_wsm, logger)
    print(f"✓ Created WSM model via factory")

    # Test GRU
    config_gru = {
        'model': {
            'model_type': 'gru',
            'time_as_channel': True,
            'forcing_prob': 1.0,
            'std_lower_bound': 0.01,
            'rnn_hidden_size': 64
        },
        'training': {
            'use_nll_loss': True
        }
    }

    model_gru = make_model(data_size, nb_classes, config_gru, logger)
    print(f"✓ Created GRU model via factory")

    # Test LSTM
    config_lstm = {
        'model': {
            'model_type': 'lstm',
            'time_as_channel': True,
            'forcing_prob': 1.0,
            'std_lower_bound': 0.01,
            'rnn_hidden_size': 64
        },
        'training': {
            'use_nll_loss': True
        }
    }

    model_lstm = make_model(data_size, nb_classes, config_lstm, logger)
    print(f"✓ Created LSTM model via factory")

    # Test FFNN
    config_ffnn = {
        'model': {
            'model_type': 'ffnn'
        },
        'training': {}
    }

    model_ffnn = make_model(data_size, nb_classes, config_ffnn, logger)
    print(f"✓ Created FFNN model via factory")


def test_wsm_tbptt():
    """Test WSM with Truncated Backpropagation Through Time."""
    print_test_header("Test 14: WSM TBPTT")

    batch_size, seq_len, data_size = 4, 40, 2

    model = WSM(
        data_size=data_size,
        width_size=32,
        depth=2,
        activation="relu",
        final_activation="tanh",
        predict_uncertainty=True,
        time_as_channel=True,
        forcing_prob=1.0,
        autoregressive_train=False,
        stochastic_ar=False
    )

    xs = torch.randn(batch_size, seq_len, data_size)
    ts = torch.linspace(0, 1, seq_len).unsqueeze(0).unsqueeze(-1).expand(batch_size, seq_len, 1)

    output = model.tbptt_non_ar_call(xs, ts, num_chunks=4)

    expected_shape = (batch_size, seq_len, data_size * 2)
    assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
    print(f"✓ TBPTT: Input {xs.shape} -> Output {output.shape}")


def test_device_compatibility():
    """Test model works on different devices."""
    print_test_header("Test 15: Device Compatibility")

    batch_size, seq_len, data_size = 2, 10, 2

    model = WSM(
        data_size=data_size,
        width_size=16,
        depth=2,
        activation="relu",
        predict_uncertainty=False,
        autoregressive_train=False,
        stochastic_ar=False
    )

    # Test on CPU
    xs_cpu = torch.randn(batch_size, seq_len, data_size)
    ts_cpu = torch.linspace(0, 1, seq_len).unsqueeze(0).unsqueeze(-1).expand(batch_size, seq_len, 1)

    output_cpu = model(xs_cpu, ts_cpu)
    print(f"✓ CPU: Output {output_cpu.shape}")

    # Test on GPU if available
    if torch.cuda.is_available():
        model_gpu = model.cuda()
        xs_gpu = xs_cpu.cuda()
        ts_gpu = ts_cpu.cuda()

        output_gpu = model_gpu(xs_gpu, ts_gpu)
        print(f"✓ CUDA: Output {output_gpu.shape}")
    else:
        print(f"⊘ CUDA not available, skipping GPU test")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*70)
    print("  PYTORCH MODELS TEST SUITE")
    print("="*70)

    tests = [
        test_root_mlp_regressor,
        test_root_mlp_classif,
        test_gradual_mlp,
        test_conv_net_1d,
        test_wsm_non_autoregressive,
        test_wsm_autoregressive,
        test_wsm_stochastic_ar,
        test_wsm_with_conv_embedding,
        test_wsm_classification,
        test_gru_model,
        test_lstm_model,
        test_ffnn_model,
        test_make_model_factory,
        test_wsm_tbptt,
        test_device_compatibility
    ]

    passed = 0
    failed = 0

    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"\n❌ TEST FAILED: {test_fn.__name__}")
            print(f"   Error: {str(e)}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "="*70)
    print(f"  TEST SUMMARY")
    print("="*70)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")

    if failed == 0:
        print("\n🎉 All tests passed! 🎉")
    else:
        print(f"\n⚠️  {failed} test(s) failed")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
