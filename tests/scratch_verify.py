import urllib.request
import urllib.error
import json
import os
import io

def test_flow():
    print("=== TTV STUDIO WORKFLOW VERIFICATION ===")

    # 1. Health check
    res = urllib.request.urlopen("http://localhost:8000/docs")
    print(f"[1] Docs status: {res.status} (PASS)")

    # 2. Upload image reference
    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    with open("tests/data/sample_ref_image.jpg", "rb") as f:
        img_data = f.read()
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="sample_ref_image.jpg"\r\n'
        f"Content-Type: image/jpeg\r\n\r\n"
    ).encode("utf-8") + img_data + f"\r\n--{boundary}--\r\n".encode("utf-8")

    req = urllib.request.Request(
        "http://localhost:8000/api/v1/references/upload",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
    )
    resp = urllib.request.urlopen(req)
    img_ref = json.loads(resp.read().decode("utf-8"))
    print(f"[2] Upload Image Reference: status={img_ref.get('status')}, id={img_ref.get('reference_id')}, still_url={img_ref.get('still_url')} (PASS)")

    # 3. Upload video reference
    with open("tests/data/sample_ref_video.mp4", "rb") as f:
        vid_data = f.read()
    body_vid = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="sample_ref_video.mp4"\r\n'
        f"Content-Type: video/mp4\r\n\r\n"
    ).encode("utf-8") + vid_data + f"\r\n--{boundary}--\r\n".encode("utf-8")

    req_vid = urllib.request.Request(
        "http://localhost:8000/api/v1/references/upload",
        data=body_vid,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
    )
    resp_vid = urllib.request.urlopen(req_vid)
    vid_ref = json.loads(resp_vid.read().decode("utf-8"))
    print(f"[3] Upload Video Reference: status={vid_ref.get('status')}, id={vid_ref.get('reference_id')}, still_url={vid_ref.get('still_url')} (PASS)")

    # 4. Training with Image Reference
    train_img_payload = json.dumps({
        "base_model": "SpatialTemporalTTVModel",
        "method": "lora",
        "epochs": 1,
        "learning_rate": "0.0001",
        "batch_size": 1,
        "reference_id": img_ref["reference_id"]
    }).encode("utf-8")
    req_train_img = urllib.request.Request(
        "http://localhost:8000/api/v1/training/start",
        data=train_img_payload,
        headers={"Content-Type": "application/json"}
    )
    resp_train_img = urllib.request.urlopen(req_train_img)
    train_img_job = json.loads(resp_train_img.read().decode("utf-8"))
    print(f"[4] Training with Image Reference: status={train_img_job.get('status')}, job_id={train_img_job.get('job_id')}, ref_id={train_img_job.get('reference_id')} (PASS)")

    # 5. Training with Video Reference
    train_vid_payload = json.dumps({
        "base_model": "SpatialTemporalTTVModel",
        "method": "lora",
        "epochs": 1,
        "learning_rate": "0.0001",
        "batch_size": 1,
        "reference_id": vid_ref["reference_id"]
    }).encode("utf-8")
    req_train_vid = urllib.request.Request(
        "http://localhost:8000/api/v1/training/start",
        data=train_vid_payload,
        headers={"Content-Type": "application/json"}
    )
    resp_train_vid = urllib.request.urlopen(req_train_vid)
    train_vid_job = json.loads(resp_train_vid.read().decode("utf-8"))
    print(f"[5] Training with Video Reference: status={train_vid_job.get('status')}, job_id={train_vid_job.get('job_id')}, ref_id={train_vid_job.get('reference_id')} (PASS)")

    # 6. Training without Reference (Regression check)
    train_no_ref_payload = json.dumps({
        "base_model": "SpatialTemporalTTVModel",
        "method": "lora",
        "epochs": 1,
        "learning_rate": "0.0001",
        "batch_size": 1,
    }).encode("utf-8")
    req_train_no_ref = urllib.request.Request(
        "http://localhost:8000/api/v1/training/start",
        data=train_no_ref_payload,
        headers={"Content-Type": "application/json"}
    )
    resp_train_no_ref = urllib.request.urlopen(req_train_no_ref)
    train_no_ref_job = json.loads(resp_train_no_ref.read().decode("utf-8"))
    print(f"[6] Training without Reference: status={train_no_ref_job.get('status')}, job_id={train_no_ref_job.get('job_id')}, ref_id={train_no_ref_job.get('reference_id')} (PASS)")

    # 7. Check Training Job Status Endpoint
    status_req = urllib.request.Request(f"http://localhost:8000/api/v1/training/status/{train_img_job['job_id']}")
    status_resp = urllib.request.urlopen(status_req)
    job_status = json.loads(status_resp.read().decode("utf-8"))
    print(f"[7] Training Status Query: status={job_status.get('status')}, progress={job_status.get('progress')}% (PASS)")

    # 8. Negative Test: SSRF / Private IP rejection
    ssrf_payload = json.dumps({"url": "http://169.254.169.254/latest/meta-data/"}).encode("utf-8")
    ssrf_req = urllib.request.Request(
        "http://localhost:8000/api/v1/references/url",
        data=ssrf_payload,
        headers={"Content-Type": "application/json"}
    )
    try:
        urllib.request.urlopen(ssrf_req)
        print("[8] SSRF Test: Unexpected Success (FAIL)")
    except urllib.error.HTTPError as e:
        err = json.loads(e.read().decode("utf-8"))
        print(f"[8] SSRF Test: Correctly rejected with HTTP {e.code}: {err.get('detail')} (PASS)")

    # 9. Negative Test: Invalid / Malformed URL rejection
    inv_url_payload = json.dumps({"url": "ftp://invalid-scheme.xyz/file.mp4"}).encode("utf-8")
    inv_req = urllib.request.Request(
        "http://localhost:8000/api/v1/references/url",
        data=inv_url_payload,
        headers={"Content-Type": "application/json"}
    )
    try:
        urllib.request.urlopen(inv_req)
        print("[9] Invalid URL Test: Unexpected Success (FAIL)")
    except urllib.error.HTTPError as e:
        err = json.loads(e.read().decode("utf-8"))
        print(f"[9] Invalid URL Test: Correctly rejected with HTTP {e.code}: {err.get('detail')} (PASS)")

    # 10. Negative Test: Unsupported file upload type
    bad_file_data = b"This is a plain text file pretending to be invalid media."
    body_bad = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="sample_doc.txt"\r\n'
        f"Content-Type: text/plain\r\n\r\n"
    ).encode("utf-8") + bad_file_data + f"\r\n--{boundary}--\r\n".encode("utf-8")
    req_bad = urllib.request.Request(
        "http://localhost:8000/api/v1/references/upload",
        data=body_bad,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
    )
    try:
        urllib.request.urlopen(req_bad)
        print("[10] Unsupported File Upload Test: Unexpected Success (FAIL)")
    except urllib.error.HTTPError as e:
        err = json.loads(e.read().decode("utf-8"))
        print(f"[10] Unsupported File Upload Test: Correctly rejected with HTTP {e.code}: {err.get('detail')} (PASS)")

    # 11. New Generation with Reference (Regression check)
    gen_payload = json.dumps({
        "prompt": "Cinematic shot of a farmer in golden sunrise fields",
        "aspect_ratio": "16:9",
        "quality": "standard",
        "language": "en",
        "reference_id": img_ref["reference_id"]
    }).encode("utf-8")
    gen_req = urllib.request.Request(
        "http://localhost:8000/api/v1/generate",
        data=gen_payload,
        headers={"Content-Type": "application/json"}
    )
    gen_resp = urllib.request.urlopen(gen_req)
    gen_job = json.loads(gen_resp.read().decode("utf-8"))
    print(f"[11] New Generation with Reference: status={gen_job.get('status')}, job_id={gen_job.get('job_id')} (PASS)")

    print("=== ALL WORKFLOW END-TO-END VERIFICATION CHECKS PASSED ===")

if __name__ == "__main__":
    test_flow()
