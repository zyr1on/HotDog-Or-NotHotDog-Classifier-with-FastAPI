# API Reference

## Base URL

```
http://localhost:8000
```

---

## Endpoints

### `GET /`

Returns the self-contained HTML frontend for uploading and classifying images in the browser. No parameters required.

---

### `POST /predict`

Accepts a multipart image upload and returns a classification result.

**Request**

```
Content-Type: multipart/form-data
Body: file=<image file>
```

Supported formats: JPEG, PNG, WebP, BMP — any format readable by Pillow.

**Response**

```json
{"label":"HOT DOG","score":0.018630878999829292,"confidence":"96%"}
```

| Field | Type | Description |
|---|---|---|
| `label` | string | `"HOT DOG"` or `"NOT HOT DOG"` |
| `score` | float | Raw sigmoid output in range [0.0, 1.0] |
| `confidence` | int | confidence of model result |


**Score interpretation**

| Score range | Label |
|---|---|
| 0.0 — 0.499 | HOT DOG |
| 0.500 — 1.0 | NOT HOT DOG |

The score reflects model confidence. A score of `0.03` is a high-confidence hotdog. A score of `0.97` is a high-confidence negative.

---

## Example

**cURL**

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@hotdog.jpg"
```

**Python**

```python
import requests

with open("hotdog.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/predict",
        files={"file": f}
    )

result = response.json()
print(result["label"])   # HOT DOG
print(result["score"])   # 0.1834
```

**JavaScript (fetch)**

```javascript
const formData = new FormData();
formData.append("file", fileInput.files[0]);

const response = await fetch("http://localhost:8000/predict", {
  method: "POST",
  body: formData,
});

const data = await response.json();
console.log(data.label, data.score);
```

---

## Error Handling

The API does not currently return structured error responses. If the uploaded file cannot be read as an image, the server will return HTTP `500`. Ensure the uploaded file is a valid image before sending.
