#include "pixconv.h"

void conv_0rgb1555_argb8888(const struct scaler_ctx *ctx, const void *input_)
{
   const uint16_t *input = (const uint16_t*)input_;
   uint32_t *output = ctx->input.frame;

   for (int h = 0; h < ctx->in_height; h++, output += ctx->input.stride >> 2, input += ctx->in_stride >> 1)
   {
      for (int w = 0; w < ctx->in_width; w++)
      {
         uint32_t col = input[w];
         uint32_t r = (col >> 10) & 0x1f;
         uint32_t g = (col >>  5) & 0x1f;
         uint32_t b = (col >>  0) & 0x1f;
         r = (r << 3) | (r >> 2);
         g = (g << 3) | (g >> 2);
         b = (b << 3) | (b >> 2);

         output[w] = (0xff << 24) | (r << 16) | (g << 8) | (b << 0);
      }
   }
}

void conv_bgr24_argb8888(const struct scaler_ctx *ctx, const void *input_)
{
   const uint8_t *input = (const uint8_t*)input_;
   uint32_t *output = ctx->input.frame;

   for (int h = 0; h < ctx->in_height; h++, output += ctx->input.stride >> 2, input += ctx->in_stride)
   {
      const uint8_t *inp = input;
      for (int w = 0; w < ctx->in_width; w++)
      {
         uint32_t b = *inp++;
         uint32_t g = *inp++;
         uint32_t r = *inp++;
         output[w] = (0xff << 24) | (r << 16) | (g << 8) | (b << 0);
      }
   }
}

void conv_argb8888_0rgb1555(const struct scaler_ctx *ctx, void *output_)
{
   const uint32_t *input = ctx->output.frame;
   uint16_t *output = (uint16_t*)output_;

   for (int h = 0; h < ctx->out_height; h++, output += ctx->out_stride >> 1, input += ctx->output.stride >> 2)
   {
      for (int w = 0; w < ctx->out_width; w++)
      {
         uint32_t col = input[w];
         uint16_t r = (col >> 19) & 0x1f;
         uint16_t g = (col >> 11) & 0x1f;
         uint16_t b = (col >>  3) & 0x1f;
         output[w] = (r << 10) | (g << 5) | (b << 0);
      }
   }
}

void conv_argb8888_bgr24(const struct scaler_ctx *ctx, void *output_)
{
   const uint32_t *input = ctx->output.frame;
   uint8_t *output = (uint8_t*)output_;

   for (int h = 0; h < ctx->out_height; h++, output += ctx->out_stride, input += ctx->output.stride >> 2)
   {
      uint8_t *out = output;
      for (int w = 0; w < ctx->out_width; w++)
      {
         uint32_t col = input[w];
         *out++ = (uint8_t)(col >>  0);
         *out++ = (uint8_t)(col >>  8);
         *out++ = (uint8_t)(col >> 16);
      }
   }
}

