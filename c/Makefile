CC      = gcc
# ARCH=native -> ottimizzato per QUESTA macchina (default, piu' veloce).
# ARCH=x86-64-v3 -> binario PORTABILE su qualsiasi x86-64 moderno con AVX2 (per distribuire).
# ARCH=x86-64 -> massima compatibilita' (niente AVX2: usa il path scalare di fallback).
ARCH   ?= native
CFLAGS  = -O3 -march=$(ARCH) -fopenmp -Wall -Wextra -Wno-unused-parameter -Wno-misleading-indentation -Wno-unused-function
LDFLAGS = -lm -fopenmp

all: glm

glm: glm.c st.h json.h tok.h tok_unicode.h
	$(CC) $(CFLAGS) glm.c -o glm $(LDFLAGS)

olmoe: olmoe.c st.h json.h
	$(CC) $(CFLAGS) olmoe.c -o olmoe $(LDFLAGS)

# binario portabile da distribuire su altre macchine x86-64
portable:
	$(MAKE) glm ARCH=x86-64-v3

clean:
	rm -f olmoe glm
