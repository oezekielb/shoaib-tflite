#ifndef SIGNAL_QUEUE_H
#define SIGNAL_QUEUE_H

#define WINDOW 128

class Queue {
  public:
    Queue() : head(0), size(0) {}

    void push(float ax, float ay, float az, float gx, float gy, float gz) {
      if (is_full()) {
        size--;
        head = (head + 1) % WINDOW;
      }

      uint tail = (head + size) % WINDOW;
      ax_buffer[tail] = ax;
      ay_buffer[tail] = ay;
      az_buffer[tail] = az;
      gx_buffer[tail] = gx;
      gy_buffer[tail] = gy;
      gz_buffer[tail] = gz;
      size++;
    }

    void pop(float &ax_out, float &ay_out, float &az_out, float &gx_out, float &gy_out, float &gz_out) {
      if (size == 0) { return; }

      ax_out = ax_buffer[head];
      ay_out = ay_buffer[head];
      az_out = az_buffer[head];
      gx_out = gx_buffer[head];
      gy_out = gy_buffer[head];
      gz_out = gz_buffer[head];
      head = (head + 1) % WINDOW;
      size--;
    }

    void peek(float *ax_out, float *ay_out, float *az_out, float *gx_out, float *gy_out, float *gz_out) {
      for (int i = 0; i < size; i++) {
        uint idx = (head + i) % WINDOW;
        ax_out[i] = ax_buffer[idx];
        ay_out[i] = ay_buffer[idx];
        az_out[i] = az_buffer[idx];
        gx_out[i] = gx_buffer[idx];
        gy_out[i] = gy_buffer[idx];
        gz_out[i] = gz_buffer[idx];
      }
    }

    bool is_full() {
      return size >= WINDOW;
    }

    uint get_size() {
      return size;
    }

  private:
    float ax_buffer[WINDOW];
    float ay_buffer[WINDOW];
    float az_buffer[WINDOW];
    float gx_buffer[WINDOW];
    float gy_buffer[WINDOW];
    float gz_buffer[WINDOW];
    uint head;
    uint size;
};

Queue signal_queue;

#endif // SIGNAL_QUEUE_H