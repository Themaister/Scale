#include "scaler.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#ifdef __SSE2__
#include <emmintrin.h>
#endif

#define FILTER_UNITY (1 << 14)

static void *scaler_alloc(size_t elem_size, size_t size)
{
   return calloc(elem_size, size);
}

static void scaler_free(void *ptr)
{
   free(ptr);
}

static inline uint8_t clamp_8bit(int32_t col)
{
   if (col > 255)
      return 255;
   else if (col < 0)
      return 0;
   else
      return (uint8_t)col;
}

static inline uint64_t build_argb64(uint16_t a, uint16_t r, uint16_t g, uint16_t b)
{
   return ((uint64_t)a << 48) | ((uint64_t)r << 32) | ((uint64_t)g << 16) | ((uint64_t)b << 0);
}

static void scaler_argb8888_vert(const struct scaler_ctx *ctx, void *output_)
{
   const uint64_t *input = ctx->scaled.frame;
   uint32_t *output = output_;

   const int16_t *filter_vert = ctx->vert.filter;

   for (int h = 0; h < ctx->out_height; h++, filter_vert += ctx->vert.filter_stride, output += ctx->out_stride >> 2)
   {
      const uint64_t *input_base = input + ctx->vert.filter_pos[h] * (ctx->scaled.stride >> 3);

      for (int w = 0; w < ctx->out_width; w++)
      {
#ifdef __SSE2__
         __m128i res = _mm_setzero_si128();

         const uint64_t *input_base_y = input_base + w;

         size_t y;
         for (y = 0; (y + 1) < ctx->vert.filter_len; y += 2, input_base_y += (ctx->scaled.stride >> 2))
         {
            __m128i coeff = _mm_set_epi64x(filter_vert[y + 1] * 0x0001000100010001ll, filter_vert[y + 0] * 0x0001000100010001ll);
            __m128i col   = _mm_set_epi64x(input_base_y[ctx->scaled.stride >> 3], input_base_y[0]);

            res = _mm_adds_epi16(_mm_mulhi_epi16(col, coeff), res);
         }

         for (; y < ctx->vert.filter_len; y++, input_base_y += (ctx->scaled.stride >> 3))
         {
            __m128i coeff = _mm_set_epi64x(0, filter_vert[y] * 0x0001000100010001ll);
            __m128i col   = _mm_set_epi64x(0, input_base_y[0]);

            res = _mm_adds_epi16(_mm_mulhi_epi16(col, coeff), res);
         }

         res = _mm_adds_epi16(_mm_srli_si128(res, 8), res);
         res = _mm_srai_epi16(res, (7 - 2 - 2));

         __m128i final = _mm_packus_epi16(res, res);

         output[w] = _mm_cvtsi128_si32(final);
#else
         int16_t res_a = 0;
         int16_t res_r = 0;
         int16_t res_g = 0;
         int16_t res_b = 0;

         const uint64_t *input_base_y = input_base + w;
         for (size_t y = 0; y < ctx->vert.filter_len; y++, input_base_y += (ctx->scaled.stride >> 3))
         {
            uint64_t col = *input_base_y;

            int16_t a = (col >> 48) & 0xffff;
            int16_t r = (col >> 32) & 0xffff;
            int16_t g = (col >> 16) & 0xffff;
            int16_t b = (col >>  0) & 0xffff;

            int16_t coeff = filter_vert[y];

            res_a += (a * coeff) >> 16;
            res_r += (r * coeff) >> 16;
            res_g += (g * coeff) >> 16;
            res_b += (b * coeff) >> 16;
         }

         res_a >>= (7 - 2 - 2);
         res_r >>= (7 - 2 - 2);
         res_g >>= (7 - 2 - 2);
         res_b >>= (7 - 2 - 2);

         output[w] = (clamp_8bit(res_a) << 24) | (clamp_8bit(res_r) << 16) | (clamp_8bit(res_g) << 8) | (clamp_8bit(res_b) << 0);
#endif
      }
   }
}

static void scaler_argb8888_horiz(const struct scaler_ctx *ctx, const void *input_)
{
   const uint32_t *input = input_;
   uint64_t *output      = ctx->scaled.frame;

   for (int h = 0; h < ctx->scaled.height; h++, input += ctx->in_stride >> 2, output += ctx->scaled.stride >> 3)
   {
      const int16_t *filter_horiz = ctx->horiz.filter;

      for (int w = 0; w < ctx->scaled.width; w++, filter_horiz += ctx->horiz.filter_stride)
      {
#ifdef __SSE2__
         __m128i res = _mm_setzero_si128();

         const uint32_t *input_base_x = input + ctx->horiz.filter_pos[w];

         size_t x;
         for (x = 0; (x + 1) < ctx->horiz.filter_len; x += 2)
         {
            __m128i coeff = _mm_set_epi64x(filter_horiz[x + 1] * 0x0001000100010001ll, filter_horiz[x + 0] * 0x0001000100010001ll);

            __m128i col = _mm_unpacklo_epi8(_mm_set_epi64x(0,
                     ((uint64_t)input_base_x[x + 1] << 32) | input_base_x[x + 0]), _mm_setzero_si128());

            col = _mm_slli_epi16(col, 7);
            res = _mm_adds_epi16(_mm_mulhi_epi16(col, coeff), res);
         }


         for (; x < ctx->horiz.filter_len; x++)
         {
            __m128i coeff = _mm_set_epi64x(0, filter_horiz[x] * 0x0001000100010001ll);
            __m128i col   = _mm_unpacklo_epi8(_mm_set_epi32(0, 0, 0, input_base_x[x]), _mm_setzero_si128());

            col = _mm_slli_epi16(col, 7);
            res = _mm_adds_epi16(_mm_mulhi_epi16(col, coeff), res);
         }

         res       = _mm_adds_epi16(_mm_srli_si128(res, 8), res);
         output[w] = _mm_cvtsi128_si64(res);
#else
         const uint32_t *input_base_x = input + ctx->horiz.filter_pos[w];

         int16_t res_a = 0;
         int16_t res_r = 0;
         int16_t res_g = 0;
         int16_t res_b = 0;

         for (size_t x = 0; x < ctx->horiz.filter_len; x++)
         {
            uint32_t col = input_base_x[x];

            int16_t a = (col >> (24 - 7)) & (0xff << 7);
            int16_t r = (col >> (16 - 7)) & (0xff << 7);
            int16_t g = (col >> ( 8 - 7)) & (0xff << 7);
            int16_t b = (col << ( 0 + 7)) & (0xff << 7);

            int16_t coeff = filter_horiz[x];

            res_a += (a * coeff) >> 16;
            res_r += (r * coeff) >> 16;
            res_g += (g * coeff) >> 16;
            res_b += (b * coeff) >> 16;
         }

         output[w] = build_argb64(res_a, res_r, res_g, res_b);
#endif
      }
   }
}

static bool allocate_filters(struct scaler_ctx *ctx)
{
   ctx->horiz.filter     = scaler_alloc(sizeof(int16_t), ctx->horiz.filter_stride * ctx->out_width);
   ctx->horiz.filter_pos = scaler_alloc(sizeof(int), ctx->out_width);

   ctx->vert.filter      = scaler_alloc(sizeof(int16_t), ctx->vert.filter_stride * ctx->out_height);
   ctx->vert.filter_pos  = scaler_alloc(sizeof(int), ctx->out_height);

   ctx->scaled.stride = ((ctx->out_width + 7) & ~7) * sizeof(uint64_t);
   ctx->scaled.width  = ctx->out_width;
   ctx->scaled.height = ctx->in_height;
   ctx->scaled.frame  = scaler_alloc(sizeof(uint64_t), (ctx->scaled.stride * ctx->scaled.height) >> 3);

   return true;
}

static bool gen_filter_point(struct scaler_ctx *ctx)
{
   ctx->horiz.filter_len    = 1;
   ctx->horiz.filter_stride = 1;
   ctx->vert.filter_len     = 1;
   ctx->vert.filter_stride  = 1;

   if (!allocate_filters(ctx))
      return false;

   int x_pos  = (1 << 15) * ctx->in_width / ctx->out_width - (1 << 15);
   int x_step = (1 << 16) * ctx->in_width / ctx->out_width;
   int y_pos  = (1 << 15) * ctx->in_height / ctx->out_height - (1 << 15);
   int y_step = (1 << 16) * ctx->in_height / ctx->out_height;

   for (int i = 0; i < ctx->out_width; i++, x_pos += x_step)
   {
      ctx->horiz.filter_pos[i] = x_pos >> 16;
      ctx->horiz.filter[i]     = FILTER_UNITY;
   }

   for (int i = 0; i < ctx->out_height; i++, y_pos += y_step)
   {
      ctx->vert.filter_pos[i] = y_pos >> 16;
      ctx->vert.filter[i]     = FILTER_UNITY;
   }
   
   return true;
}

static bool gen_filter_bilinear(struct scaler_ctx *ctx)
{
   ctx->horiz.filter_len    = 2;
   ctx->horiz.filter_stride = 2;
   ctx->vert.filter_len     = 2;
   ctx->vert.filter_stride  = 2;

   if (!allocate_filters(ctx))
      return false;

   int x_pos  = (1 << 15) * ctx->in_width / ctx->out_width - (1 << 15);
   int x_step = (1 << 16) * ctx->in_width / ctx->out_width;
   int y_pos  = (1 << 15) * ctx->in_height / ctx->out_height - (1 << 15);
   int y_step = (1 << 16) * ctx->in_height / ctx->out_height;

   for (int i = 0; i < ctx->out_width; i++, x_pos += x_step)
   {
      ctx->horiz.filter_pos[i]     = x_pos >> 16;
      ctx->horiz.filter[i * 2 + 1] = (x_pos & 0xffff) >> 2;
      ctx->horiz.filter[i * 2 + 0] = FILTER_UNITY - ctx->horiz.filter[i * 2 + 1];
   }

   for (int i = 0; i < ctx->out_height; i++, y_pos += y_step)
   {
      ctx->vert.filter_pos[i]     = y_pos >> 16;
      ctx->vert.filter[i * 2 + 1] = (y_pos & 0xffff) >> 2;
      ctx->vert.filter[i * 2 + 0] = FILTER_UNITY - ctx->vert.filter[i * 2 + 1];
   }

   return true;
}

static inline double sinc(double phase)
{
   if (fabs(phase) < 0.0001)
      return 1.0;
   else
      return sin(phase) / phase;
}

static bool gen_filter_sinc(struct scaler_ctx *ctx)
{
   ctx->horiz.filter_len    = 8;
   ctx->horiz.filter_stride = 8;
   ctx->vert.filter_len     = 8;
   ctx->vert.filter_stride  = 8;

   if (!allocate_filters(ctx))
      return false;

   int x_pos  = (1 << 15) * ctx->in_width / ctx->out_width - (1 << 15) - 3;
   int x_step = (1 << 16) * ctx->in_width / ctx->out_width;
   int y_pos  = (1 << 15) * ctx->in_height / ctx->out_height - (1 << 15) - 3;
   int y_step = (1 << 16) * ctx->in_height / ctx->out_height;

   for (int i = 0; i < ctx->out_width; i++, x_pos += x_step)
   {
      ctx->horiz.filter_pos[i] = x_pos >> 16;

      for (int j = 0; j < 8; j++)
      {
         double sinc_phase    = M_PI * ((double)(3 + (x_pos & 0xffff)) / 0x10000 - j);
         double lanczos_phase = sinc_phase / 2.0;
         int16_t sinc_val     = FILTER_UNITY * sinc(sinc_phase) * sinc(lanczos_phase);

         ctx->horiz.filter[i * 8 + j] = sinc_val;
      }
   }

   for (int i = 0; i < ctx->out_height; i++, y_pos += y_step)
   {
      ctx->vert.filter_pos[i] = y_pos >> 16;

      for (int j = 0; j < 8; j++)
      {
         double sinc_phase    = M_PI * ((double)(3 + (y_pos & 0xffff)) / 0x10000 - j);
         double lanczos_phase = sinc_phase / 2.0;
         int16_t sinc_val     = FILTER_UNITY * sinc(sinc_phase) * sinc(lanczos_phase);

         ctx->vert.filter[i * 8 + j] = sinc_val;
      }
   }

   return true;
}

static bool validate_filter(struct scaler_ctx *ctx)
{
   int max_w_pos = ctx->in_width - ctx->horiz.filter_len;
   for (int i = 0; i < ctx->out_width; i++)
   {
      if (ctx->horiz.filter_pos[i] > max_w_pos || ctx->horiz.filter_pos[i] < 0)
      {
         fprintf(stderr, "Out X = %d => In X = %d\n", i, ctx->horiz.filter_pos[i]); 
         return false;
      }
   }

   int max_h_pos = ctx->in_height - ctx->vert.filter_len;
   for (int i = 0; i < ctx->out_height; i++)
   {
      if (ctx->vert.filter_pos[i] > max_h_pos || ctx->vert.filter_pos[i] < 0)
      {
         fprintf(stderr, "Out Y = %d => In Y = %d\n", i, ctx->vert.filter_pos[i]); 
         return false;
      }
   }

   return true;
}

static void fixup_filter_sub(struct scaler_filter *filter, int out_len, int in_len)
{
   int max_pos = in_len - filter->filter_len;

   for (int i = 0; i < out_len; i++)
   {
      int postsample = filter->filter_pos[i] - max_pos;
      int presample  = -filter->filter_pos[i];

      if (postsample > 0)
      {
         filter->filter_pos[i] -= postsample;

         int16_t *base_filter = filter->filter + i * filter->filter_stride;

         if (postsample > filter->filter_len)
            memset(base_filter, 0, filter->filter_len * sizeof(int16_t));
         else
         {
            memmove(base_filter + postsample, base_filter, (filter->filter_len - postsample) * sizeof(int16_t));
            memset(base_filter, 0, postsample * sizeof(int16_t));
         }
      }

      if (presample > 0)
      {
         filter->filter_pos[i] += presample;
         int16_t *base_filter = filter->filter + i * filter->filter_stride;

         memmove(base_filter, base_filter + presample, (filter->filter_len - presample) * sizeof(int16_t));
         memset(base_filter + (filter->filter_len - presample), 0, presample * sizeof(int16_t));
      }
   }
}

// Makes sure that we never sample outside our rectangle.
static void fixup_filter(struct scaler_ctx *ctx)
{
   fixup_filter_sub(&ctx->horiz, ctx->out_width, ctx->in_width);
   fixup_filter_sub(&ctx->vert, ctx->out_height, ctx->in_height);
}

bool scaler_ctx_gen_filter(struct scaler_ctx *ctx)
{
   scaler_ctx_gen_reset(ctx);
   ctx->scaler_horiz = scaler_argb8888_horiz;
   ctx->scaler_vert  = scaler_argb8888_vert;

   bool ret = true;

   switch (ctx->scaler_type)
   {
      case SCALER_TYPE_POINT:
         ret = gen_filter_point(ctx);
         break;

      case SCALER_TYPE_BILINEAR:
         ret = gen_filter_bilinear(ctx);
         break;

      case SCALER_TYPE_SINC:
         ret = gen_filter_sinc(ctx);
         break;

      default:
         return false;
   }

   if (!ret)
      return false;

   fixup_filter(ctx);
   return validate_filter(ctx);
}

void scaler_ctx_gen_reset(struct scaler_ctx *ctx)
{
   scaler_free(ctx->horiz.filter);
   scaler_free(ctx->horiz.filter_pos);
   scaler_free(ctx->vert.filter);
   scaler_free(ctx->vert.filter_pos);
   scaler_free(ctx->scaled.frame);

   memset(&ctx->horiz, 0, sizeof(ctx->horiz));
   memset(&ctx->vert, 0, sizeof(ctx->vert));
   memset(&ctx->scaled, 0, sizeof(ctx->scaled));
}

void scaler_ctx_scale(const struct scaler_ctx *ctx,
      void *output, const void *input)
{
   ctx->scaler_horiz(ctx, input);
   ctx->scaler_vert(ctx, output);
}

