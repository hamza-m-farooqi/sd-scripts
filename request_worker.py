import time
from request_queue import job_queue
from request_processor import process_request
def viton_task_loop():
    print("worker running")
    while True:
        if not job_queue.pending:
            time.sleep(0.05)
            continue
        
        if job_queue.last_job_id is None or job_queue.last_job_id not in [job.job_id for job in job_queue.pending]:
            job_to_process = job_queue.pending[0]
            process_request(job_to_process)
        time.sleep(0.05)

# if __name__ == "__main__":
#     viton_task_loop()