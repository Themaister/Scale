TARGET := scaler

CC := clang

SOURCES := $(wildcard *.c)
OBJECTS := $(SOURCES:.c=.o)
LIBS := $(shell pkg-config imlib2 --libs) -lrt -lm
CFLAGS += $(shell pkg-config imlib2 --cflags) -O3 -ffast-math -g -Wall -pedantic -std=gnu99 -march=native

$(TARGET): $(OBJECTS)
	$(CC) -o $@ $^ $(LIBS)

%.o: %.c
	$(CC) -c -o $@ $< $(CFLAGS)

clean:
	rm -f $(TARGET) $(OBJECTS)

.PHONY: clean
