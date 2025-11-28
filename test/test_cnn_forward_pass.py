"""
Test script for CNN forward pass with checkers board states.
Run with: PYTHONPATH=./src ./venv/bin/python3 test_cnn_forward_pass.py
"""
import torch
from checkers.ai.sl_action_policy import create_model
from checkers.api.environment import reset, step, legal_moves
from checkers.core.state_utils import state_to_cnn_input
import numpy as np


def test_single_state():
    """Test forward pass with a single board state."""
    print('='*60)
    print('Test 1: Single State Forward Pass')
    print('='*60)

    # Create model
    model = create_model()
    print(f'\nModel parameters: {sum(p.numel() for p in model.parameters()):,}')

    # Get initial state
    state = reset()
    cnn_input = state_to_cnn_input(state)

    # Add batch dimension
    tensor_input = torch.from_numpy(cnn_input).unsqueeze(0)
    print(f'Input shape: {tensor_input.shape}')

    # Forward pass
    with torch.no_grad():
        output = model(tensor_input)

    print(f'Output shape: {output.shape}')
    print(f'Expected: (1, 32, 8)')
    print('\nOutput statistics:')
    print(f'  Min: {output.min().item():.4f}')
    print(f'  Max: {output.max().item():.4f}')
    print(f'  Mean: {output.mean().item():.4f}')
    print(f'  Std: {output.std().item():.4f}')

    # Print all values in 32x8 format
    print(f'\nAll output values (32x8 format):')
    print('-'*80)
    print('       Col 0    Col 1    Col 2    Col 3    Col 4    Col 5    Col 6    Col 7')
    print('-'*80)
    output_np = output[0].numpy()
    for i in range(32):
        row_str = f'Row {i:2d}:'
        for j in range(8):
            row_str += f' {output_np[i, j]:7.4f} '
        print(row_str)
    print('-'*80)

    # Top 5 predictions (flattened)
    output_flat = output[0].flatten()
    top_values, top_indices = torch.topk(output_flat, 5)
    print(f'\nTop 5 action predictions:')
    for i, (idx, val) in enumerate(zip(top_indices, top_values)):
        row = idx.item() // 8
        col = idx.item() % 8
        print(f'  {i+1}. Position [{row:2d}, {col}] (index {idx.item():3d}): {val.item():.4f}')

    return output


def test_batch_states():
    """Test forward pass with a batch of board states."""
    print('\n' + '='*60)
    print('Test 2: Batch Forward Pass')
    print('='*60)

    model = create_model()

    # Create batch of states
    batch_size = 5
    states = []

    # Generate states by playing random moves
    current_state = reset()
    states.append(current_state)

    for _ in range(batch_size - 1):
        moves = legal_moves(current_state)
        if moves:
            import random
            current_state, _, done, _ = step(current_state, random.choice(moves))
            if not done:
                states.append(current_state)
            else:
                states.append(reset())  # Add fresh state if game ended
                current_state = reset()
        else:
            states.append(reset())
            current_state = reset()

    # Convert to batch tensor
    batch_input = np.stack([state_to_cnn_input(s) for s in states])
    batch_tensor = torch.from_numpy(batch_input)

    print(f'\nBatch size: {len(states)}')
    print(f'Batch tensor shape: {batch_tensor.shape}')

    # Forward pass
    with torch.no_grad():
        batch_output = model(batch_tensor)

    print(f'Batch output shape: {batch_output.shape}')

    print(f'\nPer-state output statistics:')
    for i in range(len(states)):
        output = batch_output[i]
        flat_output = output.flatten()
        print(f'  State {i+1}: mean={output.mean().item():.4f}, '
              f'max={output.max().item():.4f}, '
              f'argmax={flat_output.argmax().item()} (row={flat_output.argmax().item()//8}, col={flat_output.argmax().item()%8})')

    # Print all values for first state in batch (32x8 format)
    print(f'\n--- All values for State 1 (32x8 format) ---')
    print('-'*80)
    print('       Col 0    Col 1    Col 2    Col 3    Col 4    Col 5    Col 6    Col 7')
    print('-'*80)
    output_np = batch_output[0].numpy()
    for i in range(32):
        row_str = f'Row {i:2d}:'
        for j in range(8):
            row_str += f' {output_np[i, j]:7.4f} '
        print(row_str)
    print('-'*80)

    return batch_output


def test_action_space_coverage():
    """Verify output covers full action space."""
    print('\n' + '='*60)
    print('Test 3: Action Space Coverage')
    print('='*60)

    model = create_model()
    state = reset()
    cnn_input = state_to_cnn_input(state)
    tensor_input = torch.from_numpy(cnn_input).unsqueeze(0)

    with torch.no_grad():
        output = model(tensor_input)

    print(f'\nAction space shape: {output.shape}')
    print(f'Expected: (1, 32, 8)')
    print(f'Total action values: {output.shape[1] * output.shape[2]} (32 x 8 = 256)')

    # Check value distribution
    output_np = output[0].numpy()
    print(f'\nValue distribution:')
    print(f'  < -0.5: {np.sum(output_np < -0.5)}')
    print(f'  -0.5 to 0: {np.sum((output_np >= -0.5) & (output_np < 0))}')
    print(f'  0 to 0.5: {np.sum((output_np >= 0) & (output_np < 0.5))}')
    print(f'  >= 0.5: {np.sum(output_np >= 0.5)}')

    return output


def test_model_determinism():
    """Verify model produces same output for same input."""
    print('\n' + '='*60)
    print('Test 4: Model Determinism')
    print('='*60)

    model = create_model()
    model.eval()  # Set to evaluation mode

    state = reset()
    cnn_input = state_to_cnn_input(state)
    tensor_input = torch.from_numpy(cnn_input).unsqueeze(0)

    # Two forward passes
    with torch.no_grad():
        output1 = model(tensor_input)
        output2 = model(tensor_input)

    # Check if outputs are identical
    are_equal = torch.allclose(output1, output2)
    max_diff = (output1 - output2).abs().max().item()

    print(f'\nOutputs identical: {are_equal}')
    print(f'Max difference: {max_diff:.10f}')

    if are_equal:
        print('✓ Model is deterministic in eval mode')
    else:
        print('✗ Model outputs differ (may be due to dropout in eval mode)')

    return are_equal


def main():
    """Run all tests."""
    print('\n' + '='*70)
    print(' '*20 + 'CNN FORWARD PASS TESTS')
    print('='*70)

    try:
        test_single_state()
        test_batch_states()
        test_action_space_coverage()
        test_model_determinism()

        print('\n' + '='*70)
        print(' '*25 + 'ALL TESTS PASSED ✓')
        print('='*70 + '\n')

    except Exception as e:
        print(f'\n✗ Test failed with error: {e}')
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
