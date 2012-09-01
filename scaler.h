#ifndef SCALER_H__
#define SCALER_H__

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

enum scaler_pix_fmt
{
   SCALER_FMT_ARGB8888 = 0
};

enum scaler_type
{
   SCALER_TYPE_UNKNOWN = 0,
   SCALER_TYPE_POINT,
   SCALER_TYPE_BILINEAR,
   SCALER_TYPE_SINC
};

struct scaler_filter
{
   int16_t *filter;
   size_t   filter_len;
   size_t   filter_stride;
   int     *filter_pos;
};

struct scaler_ctx
{
   int in_width;
   int in_height;
   int in_stride;

   int out_width;
   int out_height;
   int out_stride;

   enum scaler_pix_fmt in_fmt;
   enum scaler_pix_fmt out_fmt;
   enum scaler_type scaler_type;

   void (*scaler_horiz)(const struct scaler_ctx*,
         const void*);
   void (*scaler_vert)(const struct scaler_ctx*,
         void*);

   struct scaler_filter horiz, vert;

   struct
   {
      uint64_t *frame;
      int width;
      int height;
      int stride;
   } scaled;
};

bool scaler_ctx_gen_filter(struct scaler_ctx *ctx);
void scaler_ctx_gen_reset(struct scaler_ctx *ctx);

void scaler_ctx_scale(const struct scaler_ctx *ctx,
      void *output, const void *input);

#endif

