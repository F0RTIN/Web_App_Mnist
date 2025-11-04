##### Having a working model/neuronal network. Messing with basic Adam's HyperParameters
from tinygrad import Tensor
from tinygrad.nn.datasets import mnist
from tinygrad.nn.optim import Adam
from tinygrad.helpers import trange
import tinygrad.nn as nn
import copy

# Defining a model
class Model:
    def __init__(self):
        self.l1 = nn.Linear(784, 512)
        self.l2 = nn.Linear(512, 256)
        self.l3 = nn.Linear(256, 128)
        self.l4 = nn.Linear(128, 10)

    def __call__(self, x: Tensor) -> Tensor:
        x = x.flatten(1)
        x = self.l1(x).relu().dropout(0.15)
        x = self.l2(x).relu().dropout(0.1)
        x = self.l3(x).relu().dropout(0.2)
        return self.l4(x).log_softmax()

# Training the model
if __name__ == '__main__':
    X_train, Y_train, X_test, Y_test = mnist()

    model = Model()
    opt = Adam(
        nn.state.get_parameters(model),
        lr=0.00085,
        b1=0.9,
        b2=0.9,
        eps=1e-8,
    )

    def train_step():
        opt.zero_grad()
        samples = Tensor.randint(512, high=int(X_train.shape[0]))
        inputs = X_train[samples]
        outputs = model(inputs)
        loss = outputs.sparse_categorical_crossentropy(Y_train[samples])
        loss.backward()
        opt.step()
        return loss

    def get_test_acc():
        return (model(X_test).argmax(axis=1) == Y_test).mean() * 100

    test_acc = float('nan')

    for step in (t := trange(100)):
        with Tensor.train():
            loss = train_step().item()
        if step % 10 == 9:
            test_acc = get_test_acc().item()
        t.set_description(f"loss: {loss:4.2f} test_acc {test_acc:4.2f}%")

---------------CHANGING b2

        lr=0.001,
        b1=0.9,
        b2=0.999,
        eps=1e-8, 
LOSS=0.20 / ACCURACY=93.97%

        lr=0.001,
        b1=0.9,
        b2=0.99,
        eps=1e-8,
LOSS=0.22 / ACCURACY=94.05%

        lr=0.001,
        b1=0.9,
        b2=0.7,
        eps=1e-8, 
LOSS=0.18 / ACCURACY=94.32%

        lr=0.001,
        b1=0.9,
        b2=0.6,
        eps=1e-8,
LOSS=1.17 / ACCURACY=78.63%

        lr=0.001,
        b1=0.9,
        b2=0.79,
        eps=1e-8,
LOSS=0.15 / ACCURACY=94.86%

        lr=0.001,
        b1=0.9,
        b2=0.9, 
        eps=1e-8,
LOSS=0.11 / ACCURACY=95.25%

###### MOST EFFICIENT : b2 = 0.9 ###########
------------------- ADDING A LAYER
Before
        self.l1 = nn.Linear(784, 256)
        self.l2 = nn.Linear(256, 128)
        self.l3 = nn.Linear(128, 10)

LOSS=0.11 / ACCURACY=95.25%

After
        self.l1 = nn.Linear(784, 512)
        self.l2 = nn.Linear(512, 256)
        self.l3 = nn.Linear(256, 128)
        self.l4 = nn.Linear(128, 10)

LOSS=0.11 / ACCURACY=95.89%

The accuracy is getting down from time to time during the computation.
MAybe try to neutralize some "bad" neurones ?

----------------CHANGING lr

        lr=0.00085,
        b1=0.9,
        b2=0.9,
        eps=1e-8,
LOSS=0.14 / ACCURACY=95.88%

        lr=0.0007,
        b1=0.9,
        b2=0.9,
        eps=1e-8,
LOSS=0.11 / ACCURACY=95.90%



--------------------ADDING A DROPOUT

        x = x.flatten(1)
        x = self.l1(x).relu().dropout(0.15)
        x = self.l2(x).relu().dropout(0.1)
        x = self.l3(x).relu().dropout(0.2)
        return self.l4(x).log_softmax()

        lr=0.00085,
        b1=0.9,
        b2=0.9,
        eps=1e-8

At Best : LOSS=0.19 / ACCURACY=95.93%


I wanted to add snapshots of previous weigths of the model during its computation. Therefore, it can go back in the arborescence to take another evolutionnal path 
if the current one isn't that good

---------------------- TEST CODE WITH UP TO n-3 SNAPSHOT IN ORDER MINIMIZE LOSS. If current_loss>previous_loss, test n-1, n-2, n-3 loss and compare. Choose best SNAPSHOT

from tinygrad import Tensor
from tinygrad.nn.datasets import mnist
from tinygrad.nn.optim import Adam
from tinygrad.helpers import trange
import tinygrad.nn as nn

 ===== MODEL =====
class Model:
    def __init__(self):
        self.l1 = nn.Linear(784, 512)
        self.l2 = nn.Linear(512, 256)
        self.l3 = nn.Linear(256, 128)
        self.l4 = nn.Linear(128, 10)

    def __call__(self, x: Tensor) -> Tensor:
        x = x.flatten(1)
        x = self.l1(x).relu().dropout(0.15)
        x = self.l2(x).relu().dropout(0.1)
        x = self.l3(x).relu().dropout(0.2)
        return self.l4(x).log_softmax()

 ===== SNAPSHOT FUNCTIONS =====
def snapshot_model(model, loss):
    # Keep realized Tensors (safe on CPU/GPU)
    param_data = [p.realize() for p in nn.state.get_parameters(model)]
    return {'param_data': param_data, 'loss': loss}

def restore_model(model, snapshot):
    # Safe in-place restore for GPU/CPU
    for p, data in zip(nn.state.get_parameters(model), snapshot['param_data']):
        p.assign(data)  # safest way to copy GPU/CPU parameters

 ===== TRAINING =====
if __name__ == '__main__':
    X_train, Y_train, X_test, Y_test = mnist()
    model = Model()
    opt = Adam(nn.state.get_parameters(model),
               lr=0.00085, b1=0.9, b2=0.9, eps=1e-8)

    snapshots = []
    max_snapshots = 3
    last_loss = float('inf')
    test_acc = float('nan')

    def get_test_acc():
        return (model(X_test).argmax(axis=1) == Y_test).mean() * 100

    for step in (t := trange(100)):
        with Tensor.train():
            # Sample mini-batch
            samples = Tensor.randint(512, high=int(X_train.shape[0]))
            inputs = X_train[samples]
            targets = Y_train[samples]

            # Forward + loss
            outputs = model(inputs)
            loss = outputs.sparse_categorical_crossentropy(targets)

            # Backward
            opt.zero_grad()
            loss.backward()

            # Save snapshot
            snapshots.append(snapshot_model(model, loss.item()))
            if len(snapshots) > max_snapshots:
                snapshots.pop(0)

            # Rollback if loss increases
            if loss.item() > last_loss and snapshots:
                # Choose snapshot with lowest loss among last 3
                best_snapshot = min(snapshots, key=lambda s: s['loss'])
                if best_snapshot['loss'] < last_loss:
                    restore_model(model, best_snapshot)
                    loss = Tensor(best_snapshot['loss'])

            # Optimizer step
            opt.step()

            # Update last_loss
            last_loss = loss.item()

        if step % 10 == 9:
            test_acc = get_test_acc().item()

      t.set_description(f"loss: {loss.item():4.2f} test_acc {test_acc:4.2f}%")

####COVNET


  B = int(getenv("BATCH", 512))
  LR = float(getenv("LR", 0.02))
  LR_DECAY = float(getenv("LR_DECAY", 0.9))
  PATIENCE = float(getenv("PATIENCE", 50))

  ANGLE = float(getenv("ANGLE", 15))
  SCALE = float(getenv("SCALE", 0.1))
  SHIFT = float(getenv("SHIFT", 0.1))
  SAMPLING = SamplingMod(getenv("SAMPLING", SamplingMod.NEAREST.value))
  
  ##loss: 0.10  accuracy: 98.43%

    B = int(getenv("BATCH", 512))
  LR = float(getenv("LR", 0.02))
  LR_DECAY = float(getenv("LR_DECAY", 0.9))
  PATIENCE = float(getenv("PATIENCE", 30))

  ANGLE = float(getenv("ANGLE", 15))
  SCALE = float(getenv("SCALE", 0.1))
  SHIFT = float(getenv("SHIFT", 0.1))
  SAMPLING = SamplingMod(getenv("SAMPLING", SamplingMod.NEAREST.value))

  ##loss: 0.13  accuracy: 98.47%

    B = int(getenv("BATCH", 512))
  LR = float(getenv("LR", 0.02))
  LR_DECAY = float(getenv("LR_DECAY", 0.8))
  PATIENCE = float(getenv("PATIENCE", 50))

  ANGLE = float(getenv("ANGLE", 15))
  SCALE = float(getenv("SCALE", 0.1))
  SHIFT = float(getenv("SHIFT", 0.1))
  SAMPLING = SamplingMod(getenv("SAMPLING", SamplingMod.NEAREST.value))

  ##loss: 0.11  accuracy: 98.36%

    B = int(getenv("BATCH", 1024))
  LR = float(getenv("LR", 0.02))
  LR_DECAY = float(getenv("LR_DECAY", 0.9))
  PATIENCE = float(getenv("PATIENCE", 50))

  ANGLE = float(getenv("ANGLE", 15))
  SCALE = float(getenv("SCALE", 0.1))
  SHIFT = float(getenv("SHIFT", 0.1))
  SAMPLING = SamplingMod(getenv("SAMPLING", SamplingMod.NEAREST.value))

    Even with twice the batch size, we are still in the same scope of accuracy.
  Really long to compute, not much improvment.
  ##loss: 0.08  accuracy: 98.62%:


