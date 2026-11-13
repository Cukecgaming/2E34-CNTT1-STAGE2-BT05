# Phân Tích Cơ Chế In-Context Learning & Ứng Dụng Few-Shot Xử Lý Ngôn Ngữ Mỉa Mai Tại TechShop

Tài liệu này giải quyết bài toán phân loại cảm xúc đánh giá sản phẩm (Sentiment Analysis) tại **TechShop**, khắc phục triệt để bẫy ngữ nghĩa mỉa mai, khen đểu của khách hàng công nghệ bằng cách vận dụng cơ chế **In-Context Learning (ICL)** và kỹ thuật **Few-shot Prompting**.

---

## Phần 1: Phân Tích Cơ Chế Hoạt Động Của In-Context Learning (ICL)

### 1. Bản chất của In-Context Learning (Học trong ngữ cảnh)
**In-Context Learning (ICL)** là khả năng của các Mô hình Ngôn ngữ Lớn (LLM) trong việc học một nhiệm vụ mới hoặc thích nghi với một sắc thái ngữ cảnh đặc thù **ngay tại thời điểm suy luận (inference time)** thông qua các ví dụ mẫu được cung cấp trực tiếp trong Prompt, mà **không cần cập nhật lại trọng số (weights)** hay huấn luyện lại (fine-tuning) mô hình.

### 2. Cơ chế giúp AI "bắt sóng" ngữ cảnh mà không cần train lại model
Dựa trên kiến trúc Transformer, quá trình này diễn ra qua 3 bước kỹ thuật cốt lõi:

* **Kích hoạt không gian biểu diễn tiềm ẩn (Latent Space Activation):** Trong quá trình tiền huấn luyện (pre-training) trên hàng nghìn tỷ token, LLM đã hấp thụ vô số cấu trúc ngôn ngữ bao gồm cả văn phong châm biếm, mỉa mai công nghệ. Các ví dụ mẫu trong Prompt đóng vai trò như một **"chiếc chìa khóa" (conditioning signal)** định vị chính xác vùng kiến thức về "sắc thái mỉa mai" đang nằm ẩn trong mạng nơ-ron.
* **Cơ chế tự chú ý (Self-Attention Mechanism):** Khi đọc câu đánh giá thực tế của khách hàng, các lớp Attention trong Transformer sẽ tính toán độ tương đồng giữa câu đánh giá mới với các cặp *(Input $\rightarrow$ Output)* mẫu nằm phía trước trong cùng cửa sổ ngữ cảnh (Context Window). Mô hình nhận ra quy luật ánh xạ: *"Khen ngợi bề mặt + Thông số kỹ thuật tồi tệ = Tiêu cực (Mỉa mai)"*.
* **Học quy luật tạm thời (Transient Task Learning):** Toàn bộ sự "học" này chỉ tồn tại tạm thời trong bộ nhớ làm việc (Activation states) của phiên giao dịch hiện tại. Sau khi trả về kết quả, tham số gốc của mô hình vẫn giữ nguyên vẹn, giúp tiết kiệm chi phí hạ tầng so với việc phải huấn luyện lại mô hình.

### 3. Tại sao Zero-shot thất bại trước bẫy ngữ nghĩa mỉa mai?
* **Zero-shot** chỉ dựa vào **nghĩa hiển ngôn bề mặt (Surface Lexical Meaning)**. Khi gặp các cụm từ mang tính tích cực cao như *"đẹp mê ly"*, *"đầm tay"*, *"khỏi cần bật lò sưởi"*, trọng số xác suất của các từ này áp đảo hoàn toàn con số *"98 độ C"*, dẫn đến kết luận sai lệch là `TÍCH CỰC`.
* **Few-shot** thiết lập lại **ranh giới quyết định (Decision Boundary)** ngay trong ngữ cảnh, buộc mô hình phải đối chiếu thông số thực tế (nhiệt độ, dung lượng pin, thời gian giao hàng) với kỳ vọng tiêu chuẩn của ngành công nghệ trước khi kết luận.

---

## Phần 2: Thiết Kế Prompt Few-Shot Xử Lý Đánh Giá Mỉa Mai (TechShop)

Dưới đây là cấu trúc Prompt hoàn chỉnh sử dụng kỹ thuật **Few-shot Chain-of-Thought (CoT)** với 3 cặp ví dụ mẫu bao phủ 3 nhóm lỗi thường gặp tại TechShop: **Nhiệt độ (Tản nhiệt), Thời lượng Pin, và Thời gian giao hàng**.

```markdown
### VAI TRÒ & NHIỆM VỤ
Bạn là Chuyên viên Kiểm soát Chất lượng (QA) cao cấp tại hệ thống bán lẻ công nghệ TechShop. Nhiệm vụ của bạn là phân loại cảm xúc thực sự của khách hàng thành 1 trong 2 nhãn: **[TÍCH CỰC]** hoặc **[TIÊU CỰC]**.

### QUY TẮC PHÂN TÍCH NGỮ NGHĨA (BẮT BUỘC)
Khách hàng mua đồ công nghệ thường sử dụng lối nói mỉa mai, châm biếm (khen bề mặt nhưng chê thông số thực tế). Bạn phải phân tích logic đằng sau các thông số kỹ thuật trước khi đưa ra kết luận, tuyệt đối không chỉ dựa vào từ ngữ khen ngợi bề mặt.

---
### VÍ DỤ MẪU (FEW-SHOT EXAMPLES)

**Ví dụ 1 (Bẫy mỉa mai về Nhiệt độ / Tản nhiệt):**
- **Input:** "Laptop mỏng nhẹ sang trọng thật sự, mới mở 3 tab Chrome mà quạt kêu êm như máy bay phản lực, kê tay ấm áp như đang sưởi than tổ ong."
- **Phân tích sắc thái:** Khách dùng từ tích cực ("sang trọng", "êm", "ấm áp") nhưng thực chất đang châm biếm quạt tản nhiệt kêu quá to ("máy bay phản lực") và máy quá nóng ("sưởi than tổ ong") dù chỉ chạy tác vụ nhẹ.
- **Output:** TIÊU CỰC

**Ví dụ 2 (Bẫy mỉa mai về Thời lượng Pin):**
- **Input:** "Điện thoại xịn xò quá, giúp mình cai nghiện mạng xã hội cực hiệu quả vì sạc đầy 100% lướt TikTok đúng 45 phút là tự động sập nguồn cho mắt nghỉ ngơi!"
- **Phân tích sắc thái:** Khách khen "xịn xò", "hiệu quả" nhưng bản chất là chê bai pin tụt quá nhanh (sử dụng 45 phút đã sập nguồn là lỗi nghiêm trọng của smartphone).
- **Output:** TIÊU CỰC

**Ví dụ 3 (Bẫy mỉa mai về Thời gian giao hàng):**
- **Input:** "Dịch vụ giao hàng của TechShop đỉnh cao của sự kiên nhẫn, đặt cái tai nghe từ hồi còn độc thân mà nay sắp cưới vợ rồi vẫn thấy trạng thái đang lấy hàng."
- **Phân tích sắc thái:** Cụm từ "đỉnh cao" mang sắc thái châm biếm việc giao hàng bị trì hoãn quá lâu, gây bức xúc cho người mua.
- **Output:** TIÊU CỰC

---
### DỮ LIỆU CẦN PHÂN LOẠI (TARGET REVIEW)
- **Input:** "Máy thiết kế đẹp mê ly, cầm rất đầm tay, mùa đông ôm máy chơi game thì khỏi cần bật lò sưởi vì nhiệt độ CPU lên hẳn 98 độ C!"
- **Yêu cầu đầu ra:** Trình bày ngắn gọn phần phân tích sắc thái và chốt nhãn Output.
```

---

### Kết Quả Chạy Thử Nghiệm Với Prompt Mới

Khi áp dụng Prompt Few-shot trên cho câu đánh giá gốc của TechShop, AI đã nhận diện chính xác bẫy ngữ nghĩa:

* **Phân tích sắc thái:** Khách hàng khen ngợi thiết kế bên ngoài ("đẹp mê ly", "đầm tay"), nhưng vế sau sử dụng phép ẩn dụ mỉa mai ("khỏi cần bật lò sưởi") để phản ánh tình trạng quá nhiệt nghiêm trọng của thiết bị (CPU lên tới $98^\circ\text{C}$).
* **Output:** **TIÊU CỰC**

---

## Hướng Dẫn Nộp Bài GitHub
1. Tạo repository mới trên GitHub theo đúng quy tắc đặt tên: `[Tên Lớp]_[Môn Học]_Session02_Ex05`.
2. Tải tệp `README.md` này lên nhánh `main` của repository.
3. Sao chép đường dẫn (URL) của repository và nộp lên hệ thống Portal.