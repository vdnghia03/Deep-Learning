

# Mạng Hồi Quy (RNN)

Mạng hồi quy (RNN) được sử dụng để xử lý dữ liệu dạng chuỗi như câu trong ngôn ngữ tự nhiên hoặc dữ liệu chuỗi thời gian (time series). Điểm đặc trưng là thông tin từ các từ đứng trước (hoặc các bước thời gian trước) sẽ ảnh hưởng đến các từ đứng sau.

## 1. Biểu diễn Đầu vào (Input Representation)

Đầu tiên, các từ trong câu được chuyển đổi thành các vector biểu diễn thông qua lớp **Embedding Layer** (tầng nhúng).

*   **Quá trình:** Từ chỉ số (index) của từ trong từ điển (Vocabulary - Vocab), ta chuyển sang vector biểu diễn có số chiều cao hơn.
*   **Ví dụ:** Nếu yêu cầu mỗi từ được biểu diễn bằng vector 4 chiều, thì đầu ra của lớp embedding sẽ là một chuỗi các vector $X_t$, mỗi $X_t$ là một vector 4 chiều.

## 2. Công thức tính toán trong RNN Cell

Mạng RNN đưa thông tin từ các bước thời gian trước vào quá trình tính toán hiện tại thông qua trạng thái ẩn (Hidden State) $H$.

### A. Tính Trạng thái Ẩn (Hidden State) $H_t$

Trong một ô RNN đơn giản (Simple RNN Cell), trạng thái ẩn tại bước thời gian $t$, ký hiệu là $\mathbf{H}_t$, được tính toán dựa trên đầu vào hiện tại ($\mathbf{X}_t$) và trạng thái ẩn của bước trước ($\mathbf{H}_{t-1}$).

Công thức tính toán (trước khi áp dụng hàm kích hoạt):
$$
\text{Tính toán} = (\mathbf{X}_{t} \cdot \mathbf{W}_{XH}) + (\mathbf{H}_{t-1} \cdot \mathbf{W}_{HH}) + \mathbf{B}_{H}
$$
Trong đó:

*   $\mathbf{X}_t$: Vector biểu diễn (embedding) của từ tại bước thời gian $t$.
*   $\mathbf{H}_{t-1}$: Trạng thái ẩn của ô RNN trước đó (thông tin từ quá khứ).
*   $\mathbf{W}_{XH}$: Ma trận trọng số (weight matrix) kết nối đầu vào $\mathbf{X}_t$ với trạng thái ẩn.
*   $\mathbf{W}_{HH}$: Ma trận trọng số kết nối trạng thái ẩn trước đó $\mathbf{H}_{t-1}$ với trạng thái ẩn hiện tại.
*   $\mathbf{B}_{H}$: Vector bias (độ lệch).

**Lưu ý:** Nếu $t=1$ (từ đầu tiên của câu), trạng thái ẩn trước đó $\mathbf{H}_0$ thường được khởi tạo bằng các giá trị 0.

### B. Hàm Kích hoạt (Activation Function)

Sau khi tính toán tổng trọng số, kết quả được đưa qua một hàm kích hoạt phi tuyến tính. Trong RNN đơn giản, hàm **$\text{TanH}$** (tangent hyperbolic) thường được sử dụng.

**Công thức đầy đủ cho $\mathbf{H}_t$:**

$$\mathbf{H}_{t} = \text{TanH}((\mathbf{X}_{t} \cdot \mathbf{W}_{XH}) + (\mathbf{H}_{t-1} \cdot \mathbf{W}_{HH}) + \mathbf{B}_{H})$$

### C. Chia sẻ Trọng số (Weight Sharing)

Các ma trận trọng số $\mathbf{W}_{XH}$ và $\mathbf{W}_{HH}$ là **chung** (dùng chung) cho tất cả các bước thời gian (ô RNN) trong cùng một tầng (layer).

## 3. Kích thước Ma trận Trọng số (Parameter Dimensions)

Nếu giả định:
*   Kích thước đầu vào (Embedding Dimension) $D_x = 4$.
*   Kích thước trạng thái ẩn/đầu ra của RNN Cell (Hidden Dimension) $D_h = 3$ (3 nodes).

Kích thước của các ma trận trọng số sẽ là:

1.  **$\mathbf{W}_{XH}$:** Kích thước là $D_x \times D_h$, tức là **$4 \times 3$**.
2.  **$\mathbf{W}_{HH}$:** Kích thước là $D_h \times D_h$, tức là **$3 \times 3$**.
3.  **$\mathbf{B}_{H}$:** Kích thước là $1 \times D_h$, tức là **$1 \times 3$**.

### Tổng số Tham số (Parameters)

Tổng số tham số cho một Simple RNN Layer (dựa trên ví dụ $D_x=4, D_h=3$) là tổng số phần tử trong các ma trận trọng số và bias:

$$\text{Tổng Tham số} = (D_x \cdot D_h) + (D_h \cdot D_h) + D_h$$
$$\text{Tổng Tham số} = (4 \cdot 3) + (3 \cdot 3) + 3 = 12 + 9 + 3 = 24$$
*(Nguồn xác nhận tổng số tham số là 24 trong ví dụ này).*

## 4. Mạng RNN Nhiều Tầng (Multi-Layer RNN)

Có thể xếp chồng nhiều RNN Layer (tầng) lên nhau.

*   **Cấu trúc:** Đầu ra (Hidden State) $\mathbf{H}_t$ của Layer thứ nhất sẽ trở thành đầu vào cho Layer thứ hai tại bước thời gian $t$.
*   **Ví dụ (Tính toán $K_t$ ở Layer 2):** Layer 2 có trạng thái ẩn là $\mathbf{K}_t$. Việc tính toán $\mathbf{K}_t$ sử dụng đầu vào $\mathbf{H}_t$ (đầu ra của Layer 1) và trạng thái ẩn trước đó của Layer 2 ($\mathbf{K}_{t-1}$).
    $$\mathbf{K}_{t} = \text{TanH}((\mathbf{H}_{t} \cdot \mathbf{W}_{HK}) + (\mathbf{K}_{t-1} \cdot \mathbf{W}_{KK}) + \mathbf{B}_{K})$$
    *(Trong đó $\mathbf{W}_{HK}$ là ma trận trọng số kết nối $\mathbf{H}_t$ với $\mathbf{K}_t$, và $\mathbf{W}_{KK}$ là ma trận trọng số nội bộ của Layer 2).*
*   **Yêu cầu Return Sequence:** Khi sử dụng RNN nhiều tầng, tầng thứ nhất (hoặc các tầng trung gian) **phải** trả về tất cả các trạng thái ẩn ($\mathbf{H}_1, \mathbf{H}_2, \dots$) thay vì chỉ trạng thái cuối cùng ($\mathbf{H}_{cuối}$). Điều này được thực hiện bằng cách đặt tham số `return_sequences` bằng `True` (nếu không đặt, nó chỉ trả về trạng thái cuối cùng).

## 5. Lớp Đầu ra và Phân loại (Output and Classification)

Trong bài toán phân loại nhị phân (ví dụ: khen hay chê review phim):

*   **Layer cuối cùng:** Sử dụng một lớp Dense (lớp kết nối đầy đủ).
*   **Số node:** 1 node (vì là phân loại nhị phân).
*   **Hàm kích hoạt:** Hàm **Sigmoid**.
    $$\text{Output} = \text{Sigmoid}(\text{Weighted Sum})$$
*   **Giải thích kết quả:** Hàm Sigmoid cho ra giá trị từ 0 đến 1. Nếu giá trị lớn hơn 0.5 là "khen" (tích cực), và nhỏ hơn 0.5 là "chê" (tiêu cực).

### Chi tiết Huấn luyện (Training Details)

*   **Hàm mất mát (Loss Function):** Sử dụng **Binary Cross Entropy** (vì sử dụng hàm Sigmoid ở lớp cuối).
*   **Bộ tối ưu hóa (Optimizer):** Adam.

## 6. Hạn chế của Simple RNN

Mạng RNN đơn giản gặp phải vấn đề nghiêm trọng khi xử lý các chuỗi dài (ví dụ, câu 500 từ).

*   **Vấn đề Vanishing Gradient (Tiêu biến đạo hàm):** Khi tính toán đạo hàm (gradient) cho các trọng số ở bước đầu thông qua quá trình lan truyền ngược (Backpropagation Through Time), đạo hàm của hàm $\text{TanH}$ được nhân lặp đi lặp lại qua nhiều bước.
*   **Hậu quả:** Nếu giá trị đạo hàm nhỏ hơn 1 và được nhân lặp lại nhiều lần (ví dụ, 4 lần với câu chỉ 3 từ, hoặc 500 lần với câu 500 từ), giá trị đó sẽ tiến về 0. Điều này khiến cho mạng không thể tính được mức độ ảnh hưởng của các từ ở xa (từ đầu câu) đến kết quả cuối cùng, dẫn đến mất mát thông tin.
*   **Giải pháp:** Hạn chế này dẫn đến việc phát triển các cơ chế phức tạp hơn sau này, như cơ chế **Attention** (ví dụ trong mô hình Transformer).
