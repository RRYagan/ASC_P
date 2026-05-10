/**
 * ascp.c - High-Performance Medical Image Preprocessor
 * * Logic: AMSMIPA (Structural Detail) + CLAHE (Contrast Control)
 * Optimization: Sliding-window histograms (Memory: <60MB, Speed: O(R))
 * Target: Android/Linux Edge (Intel i5 / 1.5GB RAM constraint)
 * * Compile: gcc -O3 -march=native ascp.c -o ascp -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize2.h"

// Default Parameters (Overridden by CLI)
int TARGET_SIZE = 896;
float ALPHA = 0.5f;
int CLALHE_RADIUS = 32;
float CLIP_LIMIT = 2.0f;
static const int SCALES[] = {3, 5, 7, 9, 11};
#define NUM_SCALES (sizeof(SCALES) / sizeof(SCALES[0]))

typedef struct { float r, g, b; } rgb_t;
typedef struct { float l, a, b; } lab_t;
typedef struct { int dx, dy; } offset_t;

// D65 Reference White
static const float Xn = 0.95047f, Yn = 1.0f, Zn = 1.08883f;

static inline float lab_f(float t) {
    return (t > 0.008856f) ? cbrtf(t) : (7.787f * t + 16.0f/116.0f);
}

static inline float lab_f_inv(float t) {
    float t3 = t * t * t;
    return (t3 > 0.008856f) ? t3 : ((t - 16.0f/116.0f) / 7.787f);
}

lab_t rgb2lab(rgb_t c) {
    float r = (c.r > 0.04045f) ? powf((c.r + 0.055f)/1.055f, 2.4f) : (c.r / 12.92f);
    float g = (c.g > 0.04045f) ? powf((c.g + 0.055f)/1.055f, 2.4f) : (c.g / 12.92f);
    float b = (c.b > 0.04045f) ? powf((c.b + 0.055f)/1.055f, 2.4f) : (c.b / 12.92f);
    float x = 0.4124564f*r + 0.3575761f*g + 0.1804375f*b;
    float y = 0.2126729f*r + 0.7151522f*g + 0.0721750f*b;
    float z = 0.0193339f*r + 0.1191920f*g + 0.9503041f*b;
    return (lab_t){ 116.0f * lab_f(y/Yn) - 16.0f, 500.0f * (lab_f(x/Xn) - lab_f(y/Yn)), 200.0f * (lab_f(y/Yn) - lab_f(z/Zn)) };
}

rgb_t lab2rgb(lab_t lab) {
    float fy = (lab.l + 16.0f) / 116.0f;
    float fx = lab.a / 500.0f + fy;
    float fz = fy - lab.b / 200.0f;
    float r = 3.2404542f*(Xn*lab_f_inv(fx)) - 1.5371385f*(Yn*lab_f_inv(fy)) - 0.4985314f*(Zn*lab_f_inv(fz));
    float g = -0.9692660f*(Xn*lab_f_inv(fx)) + 1.8760108f*(Yn*lab_f_inv(fy)) + 0.0415560f*(Zn*lab_f_inv(fz));
    float b = 0.0556434f*(Xn*lab_f_inv(fx)) - 0.2040259f*(Yn*lab_f_inv(fy)) + 1.0572252f*(Zn*lab_f_inv(fz));
    return (rgb_t){ (r > 0.0031308f) ? 1.055f*powf(r, 1.0f/2.4f)-0.055f : 12.92f*r,
                    (g > 0.0031308f) ? 1.055f*powf(g, 1.0f/2.4f)-0.055f : 12.92f*g,
                    (b > 0.0031308f) ? 1.055f*powf(b, 1.0f/2.4f)-0.055f : 12.92f*b };
}

// --- Optimization: Disk Mask Morphology ---
void morphology_op(const uint8_t *src, uint8_t *dst, int w, int h, int r, int is_dilate) {
    int mask_size = 0, r2 = r * r;
    offset_t *mask = malloc(sizeof(offset_t) * (2*r+1) * (2*r+1));
    for (int dy = -r; dy <= r; dy++)
        for (int dx = -r; dx <= r; dx++)
            if (dx*dx + dy*dy <= r2) mask[mask_size++] = (offset_t){dx, dy};

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            uint8_t res = is_dilate ? 0 : 255;
            for (int i = 0; i < mask_size; i++) {
                int nx = x + mask[i].dx, ny = y + mask[i].dy;
                if (nx >= 0 && nx < w && ny >= 0 && ny < h) {
                    uint8_t val = src[ny * w + nx];
                    if (is_dilate) { if (val > res) res = val; }
                    else { if (val < res) res = val; }
                }
            }
            dst[y * w + x] = res;
        }
    }
    free(mask);
}

// --- AMSMIPA detail extraction ---
void amsmipa_l(uint8_t *L, int w, int h) {
    uint8_t *tmp = malloc(w * h), *open = malloc(w * h), *close = malloc(w * h);
    float *detail = calloc(w * h, sizeof(float));
    for (int s = 0; s < NUM_SCALES; s++) {
        int r = SCALES[s];
        morphology_op(L, tmp, w, h, r, 0); morphology_op(tmp, open, w, h, r, 1);
        morphology_op(L, tmp, w, h, r, 1); morphology_op(tmp, close, w, h, r, 0);
        for (int i = 0; i < w * h; i++) detail[i] += ((float)L[i] - open[i]) + ((float)close[i] - L[i]);
    }
    for (int i = 0; i < w * h; i++) {
        float v = L[i] + ALPHA * detail[i];
        L[i] = (uint8_t)fmaxf(0, fminf(255, v + 0.5f));
    }
    free(tmp); free(open); free(close); free(detail);
}

// --- Optimization: True Sliding Window CLAHE O(R) ---
void clalhe_l(uint8_t *L, int w, int h) {
    uint8_t *out = malloc(w * h);
    int hist[256];
    int r = CLALHE_RADIUS;
    float clip_limit = fmaxf(1.0f, CLIP_LIMIT * (2*r+1)*(2*r+1) / 256.0f);

    for (int y = 0; y < h; y++) {
        memset(hist, 0, sizeof(hist));
        int y_min = fmaxf(0, y - r), y_max = fminf(h - 1, y + r);
        
        // Initial window for the row
        for (int wy = y_min; wy <= y_max; wy++)
            for (int wx = 0; wx <= r && wx < w; wx++) hist[L[wy * w + wx]]++;

        for (int x = 0; x < w; x++) {
            // Sliding Logic: Add/Remove columns
            if (x > r) { // Remove left column
                int old_x = x - r - 1;
                for (int wy = y_min; wy <= y_max; wy++) hist[L[wy * w + old_x]]--;
            }
            if (x + r < w && x > 0) { // Add right column
                int new_x = x + r;
                for (int wy = y_min; wy <= y_max; wy++) hist[L[wy * w + new_x]]++;
            }

            int temp_hist[256], excess = 0;
            memcpy(temp_hist, hist, sizeof(hist));
            for (int i = 0; i < 256; i++) 
                if (temp_hist[i] > clip_limit) { excess += (temp_hist[i] - clip_limit); temp_hist[i] = clip_limit; }
            
            int inc = excess / 256, rem = excess % 256;
            for (int i = 0; i < 256; i++) temp_hist[i] += inc + (i < rem);

            int sum = 0, current_val = L[y * w + x];
            for (int i = 0; i <= current_val; i++) sum += temp_hist[i];
            
            int window_area = (y_max - y_min + 1) * (fminf(w-1, x+r) - fmaxf(0, x-r) + 1);
            out[y * w + x] = (uint8_t)(255.0f * sum / window_area);
        }
    }
    memcpy(L, out, w * h); free(out);
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s in.jpg out.png [alpha] [rad] [clip] [size]\n", argv[0]);
        return 1;
    }
    if (argc > 3) ALPHA = atof(argv[3]);
    if (argc > 4) CLALHE_RADIUS = atoi(argv[4]);
    if (argc > 5) CLIP_LIMIT = atof(argv[5]);
    if (argc > 6) TARGET_SIZE = atoi(argv[6]);

    int w, h, ch;
    uint8_t *img = stbi_load(argv[1], &w, &h, &ch, 3);
    if (!img) return 1;

    uint8_t *resized = malloc(TARGET_SIZE * TARGET_SIZE * 3);
    stbir_resize_uint8_linear(img, w, h, 0, resized, TARGET_SIZE, TARGET_SIZE, 0, STBIR_RGB);

    uint8_t *L = malloc(TARGET_SIZE * TARGET_SIZE), *A = malloc(TARGET_SIZE * TARGET_SIZE), *B = malloc(TARGET_SIZE * TARGET_SIZE);
    for (int i = 0; i < TARGET_SIZE * TARGET_SIZE; i++) {
        lab_t lab = rgb2lab((rgb_t){resized[i*3]/255.0f, resized[i*3+1]/255.0f, resized[i*3+2]/255.0f});
        L[i] = (uint8_t)(lab.l * 2.55f); A[i] = (uint8_t)(lab.a + 128); B[i] = (uint8_t)(lab.b + 128);
    }

    amsmipa_l(L, TARGET_SIZE, TARGET_SIZE);
    clalhe_l(L, TARGET_SIZE, TARGET_SIZE);

    for (int i = 0; i < TARGET_SIZE * TARGET_SIZE; i++) {
        rgb_t c = lab2rgb((lab_t){L[i]/2.55f, A[i]-128.0f, B[i]-128.0f});
        resized[i*3] = (uint8_t)(fmaxf(0, fminf(1, c.r)) * 255);
        resized[i*3+1] = (uint8_t)(fmaxf(0, fminf(1, c.g)) * 255);
        resized[i*3+2] = (uint8_t)(fmaxf(0, fminf(1, c.b)) * 255);
    }

    stbi_write_png(argv[2], TARGET_SIZE, TARGET_SIZE, 3, resized, 0);
    free(img); free(resized); free(L); free(A); free(B);
    return 0;
}