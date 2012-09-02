#ifndef PIXCONV_H__
#define PIXCONV_H__

#include "scaler.h"

void conv_0rgb1555_argb8888(const struct scaler_ctx *ctx, const void *input);
void conv_bgr24_argb8888(const struct scaler_ctx *ctx, const void *input);

void conv_argb8888_0rgb1555(const struct scaler_ctx *ctx, void *output);
void conv_argb8888_bgr24(const struct scaler_ctx *ctx, void *output);

#endif

