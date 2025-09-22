

# Fast R-CNN

## 1. Tổng quan và Cải tiến so với R-CNN

Mục tiêu của Fast R-CNN là làm cho mô hình R-CNN trước đó chạy nhanh hơn.

**Tóm tắt R-CNN:**
*   Đầu vào là ảnh, chạy thuật toán Selective Search để tạo ra các đề xuất vùng (region proposals).
*   Có khoảng 2.000 đề xuất vùng (kích thước $224 \times 224$).
*   Mỗi vùng ảnh được truyền qua mô-đun CNN riêng biệt để trích xuất đặc trưng.
*   Đầu ra bao gồm điểm lớp đối tượng (C+1, bao gồm cả nền) và tọa độ hộp giới hạn (bounding box coordinates).

**Ý tưởng cốt lõi của Fast R-CNN:**
Thay vì truyền 2.000 vùng ảnh qua CNN, Fast R-CNN chỉ truyền toàn bộ ảnh đầu vào ($224 \times 224$) qua mô-đun CNN *một lần* để trích xuất đặc trưng toàn ảnh. Về mặt logic, các vùng ảnh đều là một phần của ảnh đầu vào, vì vậy việc trích xuất đặc trưng một lần sẽ cho kết quả tương tự và cải thiện đáng kể thời gian chạy (runtime).

## 2. Kiến trúc Fast R-CNN

Kiến trúc hoạt động như sau:
1.  **Ảnh đầu vào** được truyền qua **Backbone CNN** (mô-đun chịu trách nhiệm trích xuất mọi đặc trưng tồn tại trong ảnh).
2.  Backbone CNN tạo ra **Đặc trưng ảnh** (Image Features).
3.  Thuật toán **Selective Search** được chạy để tạo ra các **Đề xuất vùng** (Region Proposals) trên các đặc trưng ảnh này.
4.  Các vùng đề xuất được **Cắt và Biến dạng** (crop and warp) bằng cách sử dụng một mô-đun đặc biệt.
5.  Các vùng đề xuất này được truyền qua một mô-đun CNN nhẹ, nông (shallow, low-weight CNN) hoặc chỉ là các lớp Fully Connected (FC layers), vốn không tốn nhiều chi phí tính toán như toàn bộ mô-đun CNN được sử dụng trong R-CNN.
6.  Giống như trước, mô-đun này tạo ra hai đầu ra: **điểm lớp đối tượng** và **hồi quy hộp giới hạn** (bounding box regression).

Nếu sử dụng kiến trúc ResNet-18, phần lớn mô-đun được dùng làm Backbone (chỉ tính toán một lần), và chỉ phần nông ở cuối được sử dụng cho mạng theo vùng (per-region network).

## 3. Lớp ROI Pooling (Region of Interest Pooling)

**Vấn đề:** Việc Cắt (cropping) thông thường không khả vi (not differentiable), điều này ngăn cản việc lan truyền ngược (back propagate) và điều chỉnh trọng số của Backbone CNN.

**Giải pháp:** Bài báo đề xuất sử dụng **ROI Pooling**.
*   ROI Pooling chia cửa sổ ROI có kích thước $H \times W$ thành một lưới phụ (grid of sub-windows) có kích thước $Capital H \times Capital W$ (ví dụ: $2 \times 2$), bất kể kích thước đầu vào là bao nhiêu.
*   Kích thước xấp xỉ của các cửa sổ phụ là $H / Capital H$ theo chiều cao và $W / Capital W$ theo chiều rộng.
*   Sau đó, thực hiện **Max Pooling** trên các giá trị trong mỗi cửa sổ phụ để đưa vào ô lưới đầu ra tương ứng.
*   Quá trình này được áp dụng riêng biệt cho từng kênh (channel).

## 4. Chi tiết Huấn luyện

### 4.1. Các thay đổi đối với CNN đã được huấn luyện trước (Pre-trained CNNs)
Các mạng được thử nghiệm bao gồm AlexNet, VGG\_CNN\_M\_1024, và VGG16.
1.  **Thay đổi 1:** Lớp Max Pooling trước các lớp Fully Connected (FC) được thay thế bằng **ROI Pooling**. Các lớp trước ROI Pooling là Backbone CNN, và lớp FC là mạng vùng nông.
2.  **Thay đổi 2:** Bộ phân loại (classifier) hiện tại (chỉ dùng cho phân loại) được loại bỏ. Thay vào đó, sử dụng **hai lớp FC**: một cho phân loại (dùng Softmax) và một cho hồi quy hộp giới hạn (dùng L1 loss).
3.  **Đầu vào:** Thay vì chỉ có ảnh, đầu vào còn bao gồm danh sách các hộp giới hạn Ground Truth.

### 4.2. Lấy mẫu Mini-batch (Mini-batch Sampling)
*   Sử dụng 2 ảnh cho mỗi mini-batch.
*   Selective Search cung cấp khoảng 2.000 ROIs, nhưng chỉ **64 ROIs** được lấy mẫu từ mỗi ảnh để huấn luyện.
*   **Logic lấy mẫu:**
    *   **25%** số ROIs (tức là 16 ROIs) được lấy từ các đề xuất đối tượng có mức chồng chéo IOU (Intersection over Union) với hộp giới hạn Ground Truth **tối thiểu là 0.5**.
    *   **75%** số ROIs còn lại được lấy mẫu từ các đề xuất đối tượng có IOU tối đa với Ground Truth nằm trong khoảng **từ 0.1 đến 0.5**.

### 4.3. Hàm Loss Đa nhiệm (Multitask Loss)
Vì có cả phân loại và hồi quy, mô hình sử dụng hàm loss đa nhiệm kết hợp hai loại loss.

Công thức loss tổng thể là:
$$L(p, u, t^u, v) = L_{cls}(p, u) + \mathbb{1}[u \ge 1] L_{loc}(t^u, v)$$

Trong đó:
*   $P$ là vector phân phối xác suất rời rạc của các lớp ($K+1$ lớp: $K$ đối tượng và 1 nền).
*   $U$ là vector Ground Truth cho phân loại.
*   $T$ là vector hồi quy hộp giới hạn được dự đoán.
*   $V$ là vector Ground Truth cho hồi quy.

1.  **Loss Phân loại ($L_{cls}$):**
    *   Là hàm **Cross Entropy Loss** (hay negative log likelihood của $P_u$).
    *   $L_{cls}(p, u) = -\log(P_u)$. ($P_u$ là xác suất được dự đoán cho lớp Ground Truth $U$).

2.  **Chỉ báo Iverson Bracket ($\mathbb{1}[u \ge 1]$):**
    *   Giá trị này bằng **1** nếu $U \ge 1$ (tức là có đối tượng trong vùng) và bằng **0** nếu là nền (background).
    *   Điều này đảm bảo rằng loss hồi quy chỉ được sử dụng và mô hình bị phạt khi dự đoán không chính xác **chỉ khi** vùng đó có đối tượng Ground Truth.

3.  **Loss Định vị/Hồi quy ($L_{loc}$):**
    *   Là **Smooth L1 Loss**.
    *   Smooth L1 Loss sử dụng một dạng L2 loss khi sự khác biệt giữa dự đoán và Ground Truth có độ lớn nhỏ hơn 1, và sử dụng một dạng L1 loss trong trường hợp còn lại.
    *   **Tham số $\lambda$** kiểm soát sự cân bằng giữa hai loss; trong bài báo gốc, $\lambda$ được đặt bằng 1, nghĩa là hai loss quan trọng ngang nhau.

## 5. Tính bất biến về tỷ lệ (Scale Invariance)

Mô hình cần phát hiện đối tượng bất kể tỷ lệ của chúng (dù đối tượng gần hay xa camera).

Có hai phương pháp để đảm bảo điều này:
1.  **Brute Force (Lực lượng tuyệt đối):** Mô hình tự học cách xử lý tính bất biến về tỷ lệ mà không cần can thiệp đặc biệt vào đầu vào.
2.  **Kim tự tháp Ảnh (Image Pyramids):**
    *   Đầu vào được điều chỉnh bằng cách tạo ra nhiều phiên bản ảnh với các kích thước khác nhau (tạo thành kim tự tháp).
    *   Trong quá trình huấn luyện, chỉ một trong các tỷ lệ được lấy mẫu ngẫu nhiên để tránh tăng thời gian huấn luyện lên gấp bốn lần.
    *   Tất cả các tỷ lệ chỉ được sử dụng trong quá trình kiểm thử (testing).

## 6. Tối ưu hóa Thời gian Suy luận bằng Truncated SVD

Một vấn đề trong thời gian suy luận (inference time) là các lớp Fully Connected (FC) có quá nhiều tham số, làm tăng thời gian chạy. Cụ thể, FC6 và FC7 chiếm 38.7% và 6.2% thời gian suy luận, tổng cộng là khoảng 45%.

**Phương pháp:** Truncated SVD (Phân tích Giá trị Suy biến bị Cắt bớt).

### 6.1. SVD (Singular Value Decomposition)
Mọi ma trận trọng số $W$ ($M \times N$) có thể được biểu diễn dưới dạng tích của ba ma trận: $W = U \cdot \Sigma \cdot V^T$.
*   $U$: Ma trận trực giao ($M \times M$), các cột là cơ sở trực chuẩn (orthonormal basis) cho $W$. Các cột được sắp xếp theo thứ bậc quan trọng (U1 quan trọng hơn U2, v.v.).
*   $\Sigma$: Ma trận đường chéo ($M \times N$), các mục nhập đường chéo là các giá trị suy biến (singular values). Chúng cũng được sắp xếp theo thứ bậc ($\Sigma_1$ lớn hơn $\Sigma_2$, v.v.).
*   $V^T$: Chuyển vị của ma trận trực giao $V$ ($N \times N$).

### 6.2. Truncated SVD
*   Ý tưởng là nếu giá trị suy biến $\Sigma_i$ nhỏ hơn một ngưỡng nhất định, thì cột tương ứng trên ma trận $U$ không quan trọng và có thể được làm tròn thành 0.
*   Bằng cách loại bỏ các giá trị suy biến nhỏ nhất (chỉ giữ lại $T$ giá trị quan trọng nhất), ma trận $W$ được xấp xỉ lại bằng cách sử dụng các ma trận nhỏ hơn.
*   **Giảm tham số:** Số lượng tham số giảm từ $U \times V$ (ban đầu) xuống còn $T \times U + T \times V$, hay $T(U+V)$ (sau khi cắt bớt), giúp giảm đáng kể thời gian tính toán.
*   Việc sử dụng Truncated SVD giúp giảm thời gian chạy của các lớp FC (FC6 và FC7) từ 45% xuống chỉ còn 19.2%.


---

## Code SVD 


### Ví dụ SVD cơ bản

```python
import torch

# 1. Khởi tạo Ma trận Trọng số (W) mẫu (4x3)
W = torch.arange(12., dtype=torch.float32).reshape(4, 3)
print("Ma trận W gốc (4x3):\n", W)
# Output:
# tensor([[ 0.,  1.,  2.],
#         [ 3.,  4.,  5.],
#         [ 6.,  7.,  8.],
#         [ 9., 10., 11.]])

# 2. Thực hiện SVD
# Lưu ý: PyTorch trả về Sigma (S) dưới dạng một vector các giá trị suy biến.
U, S, V = torch.linalg.svd(W) 

print("\nU (Ma trận trực giao - 4x3 vì cột cuối vô dụng):\n", U)
print("\nS (Vector giá trị suy biến):\n", S)
print("\nV (Ma trận trực giao - 3x3):\n", V)
# (Lưu ý: PyTorch/Numpy có thể không tính cột/hàng vô dụng,
# nên U có thể là 4x3 thay vì 4x4 như lý thuyết)

# 3. Chuyển đổi S thành ma trận đường chéo (Sigma)
Sigma_diag = torch.diag(S)
print("\nSigma (Ma trận đường chéo 3x3):\n", Sigma_diag)

# 4. Tái tạo Ma trận W gốc (U * Sigma * V.T)
# V.T là V chuyển vị (Transpose). Linalg.svd trả về V, không phải V.T.
W_reconstructed = U @ Sigma_diag @ V.T

print("\nW Tái tạo:\n", W_reconstructed) 
# Các giá trị rất gần với W gốc.
```

### Ví dụ Truncated SVD (Cắt bớt SVD)

Chúng ta sẽ chọn $T=2$ (chỉ giữ lại hai giá trị suy biến quan trọng nhất).

```python
import torch.nn.functional as F

T = 2  # Số lượng giá trị suy biến quan trọng nhất được giữ lại

# 1. Cắt bớt vector S (chỉ giữ lại T=2 giá trị đầu tiên)
S_truncated = S[:T]
print("\nVector S bị cắt (T=2):\n", S_truncated)

# 2. Cắt bớt U và V (chỉ giữ lại các cột/hàng tương ứng)
# U_truncated (chỉ lấy T cột đầu tiên của U)
U_truncated = U[:, :T]
# V_truncated (chỉ lấy T hàng đầu tiên của V)
V_truncated = V[:T, :]

# 3. Tạo Ma trận Đường chéo Truncated (Sigma_truncated_diag)
# Phải đệm (pad) bằng số 0 để nó khớp với kích thước ma trận W ban đầu (3x3)
# Đệm 1 cột bên phải (right) và 1 hàng bên dưới (bottom)
# Kích thước đầu ra mong muốn là 3x3. Ma trận S_truncated là 2x1. 
# Phải chuyển S_truncated thành ma trận 2x2 trước khi padding

# Chuyển S_truncated thành ma trận đường chéo
S_diag_2x2 = torch.diag(S_truncated)

# Đệm (padding) S_diag_2x2 để trở thành 3x3
# (left, right, top, bottom)
Sigma_truncated_diag = F.pad(S_diag_2x2, (0, 1, 0, 1), mode='constant', value=0)

print("\nSigma_truncated (3x3 sau khi padding):\n", Sigma_truncated_diag)
# Giá trị cuối cùng trên đường chéo là 0, biểu thị việc loại bỏ S3.

# 4. Tái tạo Ma trận W bằng Truncated SVD (U_truncated * Sigma_truncated_diag * V_truncated.T)
# Lưu ý: Trong mã ví dụ này, chúng ta sử dụng lại ma trận V gốc sau khi cắt hàng:
# Tuy nhiên, để tuân theo công thức W = U * Sigma * V.T, chúng ta dùng V_truncated.
W_truncated_reconstructed = U @ Sigma_truncated_diag @ V.T 
# Hoặc W_truncated_reconstructed = U_truncated @ S_diag_2x2 @ V_truncated.T (nếu sử dụng T cột/hàng)

# Sử dụng công thức theo nguồn: 
W_truncated_reconstructed = U_truncated @ S_diag_2x2 @ V_truncated
print("\nW Tái tạo bằng Truncated SVD:\n", W_truncated_reconstructed) 
# Kết quả rất gần với W gốc, chứng tỏ việc loại bỏ giá trị suy biến nhỏ không gây thay đổi lớn.
```
