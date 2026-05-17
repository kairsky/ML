# Manual Worked Example: k-NN Classification (k=3)

## Dataset (2D, binary)

| Point | x₁ | x₂ | Label |
|-------|----|----|-------|
| A     | 1  | 1  | Red   |
| B     | 2  | 1  | Red   |
| C     | 3  | 3  | Blue  |
| D     | 6  | 5  | Blue  |
| E     | 7  | 7  | Blue  |

**Query point Q:** (4, 2)

## Step 1: Compute Euclidean Distances

$$d(Q, P) = \sqrt{(x_1^Q - x_1^P)^2 + (x_2^Q - x_2^P)^2}$$

| Point | Calculation | Distance |
|-------|-------------|----------|
| A | √((4-1)² + (2-1)²) = √(9+1) | **3.16** |
| B | √((4-2)² + (2-1)²) = √(4+1) | **2.24** |
| C | √((4-3)² + (2-3)²) = √(1+1) | **1.41** |
| D | √((4-6)² + (2-5)²) = √(4+9) | **3.61** |
| E | √((4-7)² + (2-7)²) = √(9+25) | **5.83** |

## Step 2: Select k=3 Nearest Neighbors

Sorted by distance: **C (1.41), B (2.24), A (3.16)**

## Step 3: Majority Vote

| Neighbor | Label |
|----------|-------|
| C | Blue |
| B | Red |
| A | Red |

**Votes:** Red = 2, Blue = 1

## Prediction

$$\hat{y}(Q) = \text{Red}$$

## Distance-Weighted Variant (optional)

Weights $w_i = 1/d_i$:

- C: Blue, w = 1/1.41 = 0.71
- B: Red, w = 1/2.24 = 0.45
- A: Red, w = 1/3.16 = 0.32

Red total: 0.77, Blue total: 0.71 → **Red** (still).
