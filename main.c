#include "scaler.h"
#include <Imlib2.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>

int main(int argc, char *argv[])
{
   if (argc != 3)
   {
      fprintf(stderr, "Usage: %s <in-file> <out-file>\n", argv[0]);
      return 1;
   }

   Imlib_Image img = imlib_load_image(argv[1]);
   if (!img)
      return 1;

   imlib_context_set_image(img);

   struct scaler_ctx ctx = {
      .in_width    = imlib_image_get_width(),
      .in_height   = imlib_image_get_height(),
      .out_width   = imlib_image_get_width() / 2,
      .out_height  = imlib_image_get_height() / 2,
      .in_stride   = imlib_image_get_width() * sizeof(uint32_t),
      .out_stride  = imlib_image_get_width() / 2 * sizeof(uint32_t),
      .in_fmt      = SCALER_FMT_ARGB8888,
      .out_fmt     = SCALER_FMT_ARGB8888,
      .scaler_type = SCALER_TYPE_SINC,
   };

   assert(scaler_ctx_gen_filter(&ctx));

   uint32_t *scale_buf = calloc(sizeof(uint32_t), ctx.out_width * ctx.out_height);

   for (unsigned i = 0; i < 4; i++)
   {
      struct timespec tv[2];
      clock_gettime(CLOCK_MONOTONIC, &tv[0]);
      scaler_ctx_scale(&ctx, scale_buf, imlib_image_get_data_for_reading_only());
      clock_gettime(CLOCK_MONOTONIC, &tv[1]);

      double time_ms = (tv[1].tv_sec - tv[0].tv_sec) * 1000.0 + (tv[1].tv_nsec - tv[0].tv_nsec) / 1000000.0;
      double ns_per_pix = (1000000.0 * time_ms) / (ctx.out_width * ctx.out_height);
      printf("Time: %.3lf ms, %.3lf ns / pixel\n", time_ms, ns_per_pix);
   }

   Imlib_Image new_img = imlib_create_image_using_data(ctx.out_width, ctx.out_height,
         scale_buf);

   imlib_free_image();
   imlib_context_set_image(new_img);
   imlib_image_set_format("png");
   imlib_save_image(argv[2]);
   imlib_free_image();

   free(scale_buf);
   scaler_ctx_gen_reset(&ctx);
}

