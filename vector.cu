#include "common.h"

#define DEF_VECTOR_SIZE 10
struct vector* vector_init(int element_size)
{
	struct vector *v = calloc(1, sizeof(struct vector));
	v->arr = calloc(DEF_VECTOR_SIZE, element_size);
	v->size = 0;
	v->capacity = DEF_VECTOR_SIZE;
	v->element_size = element_size;
	return v;
}

vector_free(struct vector*)
{
	free(v->arr);
	free(v);
}


