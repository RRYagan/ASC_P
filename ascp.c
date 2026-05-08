/**
 * ascp.c
 * 
 * AMSMIPA + CLALHE pipeline for malaria/skin images.
 * 
 * Compile:
 *   gcc -O3 -march=native -o ascp ascp.c -lm
 * 
 * Usage:
 *   ./ascp input.jpg output.png
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

// ---------- Constants ----------

int TARGET_SIZE = 896;
int NUM_SCALES = 5;
float ALPHA = 0.5f;
int CLALHE_RADIUS = 32;
float CLIP_LIMIT = 2.0f;
static const int SCALES[] = {3, 5, 7, 9, 11};
#define NUM_SCALES (sizeof(SCALES) / sizeof(SCALES[0]))


typedef struct { float r, g, b; } rgb_t;
typedef struct { float l, a, b; } lab_t;
typedef struct { int dx, dy; } offset_t;

// D65 reference white
static const float Xn = 0.95047f, Yn = 1.0f, Zn = 1.08883f;

// --- LAB Conversion Logic ---
static inline float lab_f(float t) {
    return (t > 0.008856f) ? cbrtf(t) : (7.787f * t + 16.0f/116.0f);
}

static inline float lab_f_inv(float t) {
    return (t > 0.206896f) ? (t*t*t) : ((t - 16.0f/116.0f) / 7.787f);
}

lab_t rgb2lab(rgb_t c) {
    float r = (c.r > 0.04045f) ? powf((c.r + 0.055f)/1.055f, 2.4f) : (c.r / 12.92f);
    float g = (c.g > 0.04045f) ? powf((c.g + 0.055f)/1.055f, 2.4f) : (c.g / 12.92f);
    float b = (c.b > 0.04045f) ? powf((c.b + 0.055f)/1.055f, 2.4f) : (c.b / 12.92f);

    float x = 0.4124564f*r + 0.3575761f*g + 0.1804375f*b;
    float y = 0.2126729f*r + 0.7151522f*g + 0.0721750f*b;
    float z = 0.0193339f*r + 0.1191920f*g + 0.9503041f*b;

    float fy = lab_f(y / Yn);
    lab_t lab;
    lab.l = 116.0f * fy - 16.0f;
    lab.a = 500.0f * (lab_f(x / Xn) - fy);
    lab.b = 200.0f * (fy - lab_f(z / Zn));
    return lab;
}

rgb_t lab2rgb(lab_t lab) {
    float fy = (lab.l + 16.0f) / 116.0f;
    float fx = lab.a / 500.0f + fy;
    float fz = fy - lab.b / 200.0f;

    float x = Xn * lab_f_inv(fx);
    float y = Yn * lab_f_inv(fy);
    float z = Zn * lab_f_inv(fz);

    float r_lin =  3.2404542f*x - 1.5371385f*y - 0.4985314f*z;
    float g_lin = -0.9692660f*x + 1.8760108f*y + 0.0415560f*z;
    float b_lin =  0.0556434f*x - 0.2040259f*y + 1.0572252f*z;

    rgb_t c;
    c.r = (r_lin > 0.0031308f) ? 1.055f * powf(r_lin, 1.0f/2.4f) - 0.055f : 12.92f * r_lin;
    c.g = (g_lin > 0.0031308f) ? 1.055f * powf(g_lin, 1.0f/2.4f) - 0.055f : 12.92f * g_lin;
    c.b = (b_lin > 0.0031308f) ? 1.055f * powf(b_lin, 1.0f/2.4f) - 0.055f : 12.92f * b_lin;
    return c;
}

// --- Optimized Morphology with Pre-calculated Masks ---
offset_t* get_disk_mask(int radius, int *count) {
    int max_points = (2 * radius + 1) * (2 * radius + 1);
    offset_t *offsets = (offset_t*)malloc(max_points * sizeof(offset_t));
    int c = 0, r2 = radius * radius;
    for (int dy = -radius; dy <= radius; dy++) {
        for (int dx = -radius; dx <= radius; dx++) {
            if (dx*dx + dy*dy <= r2) {
                offsets[c].dx = dx; offsets[c].dy = dy; c++;
            }
        }
    }
    *count = c; return offsets;
}

static void apply_morph(const uint8_t *src, uint8_t *dst, int w, int h, offset_t *offsets, int count, int is_erode) {
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            uint8_t target = is_erode ? 255 : 0;
            for (int i = 0; i < count; i++) {
                int nx = x + offsets[i].dx, ny = y + offsets[i].dy;
                if (nx >= 0 && nx < w && ny >= 0 && ny < h) {
                    uint8_t v = src[ny * w + nx];
                    if (is_erode) { if (v < target) target = v; }
                    else { if (v > target) target = v; }
                }
            }
            dst[y * w + x] = target;
        }
    }
}

// ---------- Optimized CLALHE for Medical Detail ----------
void clalhe_l(uint8_t *L, int w, int h) {
    int rad = CLALHE_RADIUS;
    int win_area = (2*rad+1)*(2*rad+1);
    
    // Increased CLIP_LIMIT to 4.0 for sharper medical contrast
    int clip_abs = (int)(CLIP_LIMIT * win_area / 256.0f);
    if (clip_abs < 1) clip_abs = 1;

    size_t stride_v = 256;
    size_t stride_x = (w + 1) * stride_v;
    int *int_hist = (int*)calloc((h + 1) * stride_x, sizeof(int));
    
    #define IH(y, x, v) int_hist[(y) * stride_x + (x) * stride_v + (v)]

    // Build Integral Histogram
    for (int y = 0; y < h; y++) {
        int row_sum[256] = {0};
        for (int x = 0; x < w; x++) {
            row_sum[L[y * w + x]]++;
            for (int v = 0; v < 256; v++) {
                IH(y + 1, x + 1, v) = IH(y, x + 1, v) + row_sum[v];
            }
        }
    }

    uint8_t *out = (uint8_t*)malloc(w * h);
    for (int y = 0; y < h; y++) {
        int y1 = fmax(0, y - rad), y2 = fmin(h, y + rad + 1);
        for (int x = 0; x < w; x++) {
            int x1 = fmax(0, x - rad), x2 = fmin(w, x + rad + 1);
            int hist[256], excess = 0;
            
            for (int v = 0; v < 256; v++) {
                hist[v] = IH(y2, x2, v) - IH(y2, x1, v) - IH(y1, x2, v) + IH(y1, x1, v);
                if (hist[v] > clip_abs) { 
                    excess += hist[v] - clip_abs; 
                    hist[v] = clip_abs; 
                }
            }
            
            int add = excess / 256;
            int cum = 0, target_val = L[y * w + x];
            for (int v = 0; v <= target_val; v++) cum += hist[v] + add;
            
            float res = (float)cum * 255.0f / win_area;
            out[y * w + x] = (uint8_t)fminf(255.0f, res);
        }
    }
    memcpy(L, out, w * h);
    free(out); free(int_hist);
}

// ---------- Final Preprocessing with Normalization ----------

// --- AMSMIPA Pipeline ---
void amsmipa_l(uint8_t *L, int w, int h) {
    uint8_t *tmp = (uint8_t*)malloc(w * h);
    uint8_t *opened = (uint8_t*)malloc(w * h);
    uint8_t *closed = (uint8_t*)malloc(w * h);
    float *detail = (float*)calloc(w * h, sizeof(float));

    for (int s = 0; s < NUM_SCALES; s++) {
        int count, r = SCALES[s];
        offset_t *mask = get_disk_mask(r, &count);
        
        apply_morph(L, tmp, w, h, mask, count, 1); // Opening: Erode
        apply_morph(tmp, opened, w, h, mask, count, 0); // then Dilate
        apply_morph(L, tmp, w, h, mask, count, 0); // Closing: Dilate
        apply_morph(tmp, closed, w, h, mask, count, 1); // then Erode

        for (int i = 0; i < w*h; i++) detail[i] += ((float)L[i] - opened[i]) + ((float)closed[i] - L[i]);
        free(mask);
    }

    for (int i = 0; i < w*h; i++) {
        float res = (float)L[i] + ALPHA * detail[i];
        L[i] = (uint8_t)fmaxf(0, fminf(255, res + 0.5f));
    }
    free(tmp); free(opened); free(closed); free(detail);
}

// --- Main Engine ---
float* preprocess_image(const uint8_t *rgb_in, int in_w, int in_h) {
    int w = TARGET_SIZE, h = TARGET_SIZE;
    uint8_t *resized = (uint8_t*)malloc(w * h * 3);
    stbir_resize_uint8_linear(rgb_in, in_w, in_h, 0, resized, w, h, 0, STBIR_RGB);

    uint8_t *L = (uint8_t*)malloc(w * h), *A = (uint8_t*)malloc(w * h), *B = (uint8_t*)malloc(w * h);
    for (int i = 0; i < w*h; i++) {
        rgb_t c = {resized[i*3]/255.0f, resized[i*3+1]/255.0f, resized[i*3+2]/255.0f};
        lab_t lab = rgb2lab(c);
        L[i] = (uint8_t)(lab.l * 2.55f);
        A[i] = (uint8_t)(lab.a + 128.0f); 
        B[i] = (uint8_t)(lab.b + 128.0f);
    }

    amsmipa_l(L, w, h); // Sharpen details
    clalhe_l(L, w, h);  // Fix local contrast

    // Normalization Step: Ensure the brightest pixel is 255 and darkest is 0
    uint8_t min_l = 255, max_l = 0;
    for(int i=0; i<w*h; i++) {
        if(L[i] < min_l) min_l = L[i];
        if(L[i] > max_l) max_l = L[i];
    }
    float range = (float)(max_l - min_l);

    float *output = (float*)malloc(w * h * 3 * sizeof(float));
    for (int i = 0; i < w*h; i++) {
        // Stretch L to full range [0, 100]
        float norm_l = (range > 0) ? ((float)(L[i] - min_l) / range) * 100.0f : L[i] / 2.55f;
        lab_t lab = {norm_l, A[i]-128.0f, B[i]-128.0f};
        rgb_t c = lab2rgb(lab);
        output[i*3]   = fmaxf(0, fminf(1, c.r));
        output[i*3+1] = fmaxf(0, fminf(1, c.g));
        output[i*3+2] = fmaxf(0, fminf(1, c.b));
    }

    free(resized); free(L); free(A); free(B);
    return output;
}


int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, 
            "Usage: %s input.jpg output.png [alpha] [clahe_radius] [clip_limit] [target_size]\n"
            "Example: %s image.jpg out.png 0.3 64 2.5 512\n", 
            argv[0], argv[0]);
        return 1;
    }

    // Parse optional arguments
    if (argc > 3) ALPHA = atof(argv[3]);
    if (argc > 4) CLALHE_RADIUS = atoi(argv[4]);
    if (argc > 5) CLIP_LIMIT = atof(argv[5]);
    if (argc > 6) TARGET_SIZE = atoi(argv[6]);

    printf("Parameters: ALPHA=%.2f, CLALHE_RADIUS=%d, CLIP_LIMIT=%.2f, TARGET_SIZE=%d\n",
           ALPHA, CLALHE_RADIUS, CLIP_LIMIT, TARGET_SIZE);

    int w, h, ch;
    uint8_t *in = stbi_load(argv[1], &w, &h, &ch, 3);
    if (!in) {
        fprintf(stderr, "Failed to load %s\n", argv[1]);
        return 1;
    }

    float *out_f = preprocess_image(in, w, h);
    uint8_t *out_8 = (uint8_t*)malloc(TARGET_SIZE * TARGET_SIZE * 3);
    for (int i = 0; i < TARGET_SIZE * TARGET_SIZE * 3; i++) 
        out_8[i] = (uint8_t)(out_f[i] * 255.0f + 0.5f);

    stbi_write_png(argv[2], TARGET_SIZE, TARGET_SIZE, 3, out_8, 0);
    free(in); free(out_f); free(out_8);
    printf("Saved to %s\n", argv[2]);
    return 0;
}