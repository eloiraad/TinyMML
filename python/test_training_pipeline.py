import sys
import numpy as np
import cvmml_api as C

sys.path.insert(0, ".")
from nn.model import Model
from nn.layer import Linear, ReLU, Softmax
from nn.optimizer import Adam
from nn.loss import CrossEntropyLoss, MSELoss

def test_training_pipeline_classification():
    print("\n=== Testing Training Pipeline (Classification) ===")
    # XOR-like problem but with completely separable 4 quadrants
    x_data = np.array([
        [0.1, 0.1],
        [0.9, 0.9],
        [0.1, 0.9],
        [0.9, 0.1]
    ], dtype=np.float32)

    # Classes: 0, 1, 2, 3
    y_data = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)

    model = Model([
        Linear(16),
        ReLU(),
        Linear(4)
    ])
    
    # MUST build the model so weights are initialized before passing to optimizer
    model.build(list(x_data.shape))

    optimizer = Adam(model.parameters(), lr=0.1)
    loss_fn = CrossEntropyLoss()

    history = model.fit(
        x_data=x_data,
        y_data=y_data,
        epochs=100,
        batch_size=4, # full batch
        optimizer=optimizer,
        loss_fn=loss_fn,
        shuffle=True,
        metric="accuracy"
    )

    # Over 100 epochs, Adam with lr=0.1 should perfectly memorize 4 points
    final_loss = history['loss'][-1]
    final_acc = history['metric'][-1]
    
    print(f"Final Classification Loss: {final_loss:.4f}")
    print(f"Final Classification Acc:  {final_acc:.4f}")
    assert final_acc == 1.0, f"Accuracy not 100% on trivial classification set (got {final_acc})"

    # Test evaluate
    eval_loss, eval_acc = model.evaluate(x_data, y_data, loss_fn, batch_size=4, metric="accuracy")
    print(f"Eval Loss: {eval_loss:.4f}, Eval Acc: {eval_acc:.4f}")
    assert eval_acc == 1.0, "Evaluate metric mismatch"

    weights = model.get_layer_weights()
    assert "0_Linear_weight" in weights
    print(f"Successfully extracted weights dictionary with keys: {list(weights.keys())}")


def test_training_pipeline_regression():
    print("\n=== Testing Training Pipeline (Regression) ===")
    
    # Simple linear regression y = 2x + 1
    x_data = np.linspace(-1, 1, 20).reshape(-1, 1).astype(np.float32)
    y_data = np.array(2.0 * x_data + 1.0, dtype=np.float32)

    model = Model([
        Linear(1)
    ])
    
    # MUST build the model so weights are initialized before passing to optimizer
    model.build(list(x_data.shape))

    opt = Adam(model.parameters(), lr=0.1)
    loss = MSELoss()

    # Custom metric tracking mean absolute error (MAE)
    def mae_metric(preds, targets):
        return np.mean(np.abs(preds - targets))

    history = model.fit(
        x_data=x_data,
        y_data=y_data,
        epochs=50,
        batch_size=10,
        optimizer=opt,
        loss_fn=loss,
        shuffle=True,
        metric=mae_metric
    )

    final_mae = history['metric'][-1]
    print(f"Final Regression MAE: {final_mae:.4f}")
    assert final_mae < 0.2, f"Failed to fit simple line, MAE={final_mae}"

if __name__ == "__main__":
    test_training_pipeline_classification()
    test_training_pipeline_regression()
    print("\nALL TRAINING PIPELINE TESTS PASSED!")
