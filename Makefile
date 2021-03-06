TARGET := scale

SOURCES := $(wildcard *.c)
OBJECTS := $(SOURCES:.c=.o)
LIBS := $(shell pkg-config imlib2 --libs) -lm

ifeq ($(CXX_BUILD), 1)
   CFLAGS += $(shell pkg-config imlib2 --cflags) -xc++ -O3 -ffast-math -g -Wall -pedantic -std=gnu++11 -march=native -DUSE_SIMD
else
   CFLAGS += $(shell pkg-config imlib2 --cflags) -O3 -ffast-math -g -Wall -pedantic -std=gnu99 -march=native -DUSE_SIMD
endif

$(TARGET): $(OBJECTS)
	$(CC) -o $@ $^ $(LIBS)

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $< 

clean:
	rm -f $(TARGET) $(OBJECTS)

.PHONY: clean

