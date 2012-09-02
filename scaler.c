#include "scaler.h"
#include "scaler_int.h"
#include "filter.h"
#include "pixconv.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

// In case aligned allocs are needed later ...
void *scaler_alloc(size_t elem_size, size_t size)
{
   return calloc(elem_size, size);
}

void scaler_free(void *ptr)
{
   free(ptr);
}

static bool allocate_frames(struct scaler_ctx *ctx)
{
   ctx->scaled.stride = ((ctx->out_width + 7) & ~7) * sizeof(uint64_t);
   ctx->scaled.width  = ctx->out_width;
   ctx->scaled.height = ctx->in_height;
   ctx->scaled.frame  = (uint64_t*)scaler_alloc(sizeof(uint64_t), (ctx->scaled.stride * ctx->scaled.height) >> 3);
   if (!ctx->scaled.frame)
      return false;

   if (ctx->in_fmt != SCALER_FMT_ARGB8888)
   {
      ctx->input.stride = ((ctx->in_width + 7) & ~7) * sizeof(uint32_t);
      ctx->input.frame = (uint32_t*)scaler_alloc(sizeof(uint32_t), (ctx->input.stride * ctx->in_height) >> 2);
      if (!ctx->input.frame)
         return false;
   }

   if (ctx->out_fmt != SCALER_FMT_ARGB8888)
   {
      ctx->output.stride = ((ctx->out_width + 7) & ~7) * sizeof(uint32_t);
      ctx->output.frame  = (uint32_t*)scaler_alloc(sizeof(uint32_t), (ctx->output.stride * ctx->out_height) >> 2);
      if (!ctx->output.frame)
         return false;
   }

   return true;
}

static bool set_pix_conv(struct scaler_ctx *ctx)
{
   switch (ctx->in_fmt)
   {
      case SCALER_FMT_ARGB8888:
         // No need to convert :D
         break;

      case SCALER_FMT_0RGB1555:
         ctx->in_pixconv = conv_0rgb1555_argb8888;
         break;

      case SCALER_FMT_BGR24:
         ctx->in_pixconv = conv_bgr24_argb8888;
         break;

      default:
         return false;
   }

   switch (ctx->out_fmt)
   {
      case SCALER_FMT_ARGB8888:
         // No need to convert :D
         break;

      case SCALER_FMT_0RGB1555:
         ctx->out_pixconv = conv_argb8888_0rgb1555;
         break;

      case SCALER_FMT_BGR24:
         ctx->out_pixconv = conv_argb8888_bgr24;
         break;

      default:
         return false;
   }

   return true;
}

bool scaler_ctx_gen_filter(struct scaler_ctx *ctx)
{
   scaler_ctx_gen_reset(ctx);
   ctx->scaler_horiz = scaler_argb8888_horiz;
   ctx->scaler_vert  = scaler_argb8888_vert;

   if (!allocate_frames(ctx))
      return false;

   if (!set_pix_conv(ctx))
      return false;

   if (!scaler_gen_filter(ctx))
      return false;

   return true;
}

void scaler_ctx_gen_reset(struct scaler_ctx *ctx)
{
   scaler_free(ctx->horiz.filter);
   scaler_free(ctx->horiz.filter_pos);
   scaler_free(ctx->vert.filter);
   scaler_free(ctx->vert.filter_pos);
   scaler_free(ctx->scaled.frame);
   scaler_free(ctx->input.frame);
   scaler_free(ctx->output.frame);

   memset(&ctx->horiz, 0, sizeof(ctx->horiz));
   memset(&ctx->vert, 0, sizeof(ctx->vert));
   memset(&ctx->scaled, 0, sizeof(ctx->scaled));
   memset(&ctx->input, 0, sizeof(ctx->input));
   memset(&ctx->output, 0, sizeof(ctx->output));
}

void scaler_ctx_scale(const struct scaler_ctx *ctx,
      void *output, const void *input)
{
   if (ctx->in_fmt != SCALER_FMT_ARGB8888)
   {
      ctx->in_pixconv(ctx, input);
      ctx->scaler_horiz(ctx, ctx->input.frame, ctx->input.stride);
   }
   else
      ctx->scaler_horiz(ctx, input, ctx->in_stride);

   if (ctx->out_fmt != SCALER_FMT_ARGB8888)
   {
      ctx->scaler_vert(ctx, ctx->output.frame, ctx->output.stride);
      ctx->out_pixconv(ctx, output);
   }
   else
      ctx->scaler_vert(ctx, output, ctx->out_stride);
}


