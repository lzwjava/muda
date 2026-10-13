/*
 * mudac - a tiny CUDA launch-syntax preprocessor for muda.
 *
 * It reads a .cu file (or stdin) and:
 *
 *   1. extracts every __global__/__device__ function into a string constant
 *      that is compiled at run time by the muda driver (there is no nvcc);
 *   2. removes those definitions from the host C output;
 *   3. rewrites `kernel<<<grid, block, shared, stream>>>(args)` into a call
 *      to mudartLaunchByName(), taking the address of each argument in a
 *      GNU statement expression;
 *   4. replaces CUDA runtime includes with mudart.h.
 *
 * Usage:
 *   mudac input.cu [-o output.c]
 *   mudac < input.cu > output.c
 *
 * Then compile the result with a C11 (GNU) compiler and link libmuda:
 *   mudac prog.cu -o prog.c
 *   cc -std=gnu11 -Iinclude prog.c build/libmuda.a -ldl -lpthread -lm -o prog
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <ctype.h>

/* ------------------------------------------------------------------ */
/* Growable buffer                                                    */
/* ------------------------------------------------------------------ */
typedef struct { char *data; size_t len, cap; } buf;

static void binit(buf *b)
{
    b->cap = 4096;
    b->len = 0;
    b->data = (char *)malloc(b->cap);
    if (b->data)
        b->data[0] = '\0';
}

static void bneed(buf *b, size_t extra)
{
    if (b->len + extra + 1 <= b->cap)
        return;
    while (b->cap < b->len + extra + 1)
        b->cap *= 2;
    b->data = (char *)realloc(b->data, b->cap);
}

static void bput(buf *b, const char *s, size_t n)
{
    bneed(b, n);
    memcpy(b->data + b->len, s, n);
    b->len += n;
    b->data[b->len] = '\0';
}

static void bputs(buf *b, const char *s) { bput(b, s, strlen(s)); }
static void bputc(buf *b, char c) { bput(b, &c, 1); }

static void bprintf(buf *b, const char *fmt, ...)
{
    char tmp[4096];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(tmp, sizeof(tmp), fmt, ap);
    va_end(ap);
    if (n > 0)
        bput(b, tmp, (size_t)(n < (int)sizeof(tmp) ? n : (int)sizeof(tmp) - 1));
}

/* ------------------------------------------------------------------ */
/* Lexing helpers                                                     */
/* ------------------------------------------------------------------ */
static int ident_start(char c) { return isalpha((unsigned char)c) || c == '_'; }
static int ident_char(char c) { return isalnum((unsigned char)c) || c == '_'; }

static size_t skip_line_comment(const char *s, size_t n, size_t i)
{
    while (i < n && s[i] != '\n')
        i++;
    return i;
}

static size_t skip_block_comment(const char *s, size_t n, size_t i)
{
    i += 2;
    while (i + 1 < n && !(s[i] == '*' && s[i + 1] == '/'))
        i++;
    return (i + 2 <= n) ? i + 2 : n;
}

static size_t skip_string(const char *s, size_t n, size_t i, char q)
{
    i++;
    while (i < n) {
        if (s[i] == '\\' && i + 1 < n) { i += 2; continue; }
        if (s[i] == q) { i++; break; }
        i++;
    }
    return i;
}

/* Index just past a comment/string starting at i, or i if none. */
static size_t skip_noncode(const char *s, size_t n, size_t i)
{
    if (i + 1 < n && s[i] == '/' && s[i + 1] == '/')
        return skip_line_comment(s, n, i);
    if (i + 1 < n && s[i] == '/' && s[i + 1] == '*')
        return skip_block_comment(s, n, i);
    if (s[i] == '"' || s[i] == '\'')
        return skip_string(s, n, i, s[i]);
    return i;
}

static size_t skip_ws(const char *s, size_t n, size_t i)
{
    for (;;) {
        while (i < n && isspace((unsigned char)s[i]))
            i++;
        size_t j = skip_noncode(s, n, i);
        if (j != i) { i = j; continue; }
        return i;
    }
}

static size_t find_matching(const char *s, size_t n, size_t open, char oc, char cc)
{
    int depth = 0;
    size_t i = open;
    while (i < n) {
        size_t j = skip_noncode(s, n, i);
        if (j != i) { i = j; continue; }
        if (s[i] == oc) depth++;
        else if (s[i] == cc) {
            depth--;
            if (depth == 0) return i;
        }
        i++;
    }
    return (size_t)-1;
}

static size_t find_triple_gt(const char *s, size_t n, size_t i)
{
    while (i + 2 < n) {
        size_t j = skip_noncode(s, n, i);
        if (j != i) { i = j; continue; }
        if (s[i] == '>' && s[i + 1] == '>' && s[i + 2] == '>')
            return i;
        i++;
    }
    return (size_t)-1;
}

/* ------------------------------------------------------------------ */
/* Spans and comma splitting                                          */
/* ------------------------------------------------------------------ */
typedef struct { const char *s; size_t start, len; } span;

static void trim_span(span *sp)
{
    while (sp->len > 0 && isspace((unsigned char)sp->s[sp->start])) {
        sp->start++;
        sp->len--;
    }
    while (sp->len > 0 && isspace((unsigned char)sp->s[sp->start + sp->len - 1]))
        sp->len--;
}

static int split_top(const char *s, size_t start, size_t end, span *out, int max)
{
    int count = 0, depth = 0;
    size_t seg = start;
    for (size_t i = start; i <= end; i++) {
        int at_end = (i == end);
        if (!at_end) {
            char c = s[i];
            if (c == '(' || c == '[' || c == '{') depth++;
            else if (c == ')' || c == ']' || c == '}') depth--;
            else if (c == ',' && depth == 0) {
                if (count < max) { out[count].s = s; out[count].start = seg; out[count].len = i - seg; }
                count++;
                seg = i + 1;
                continue;
            }
        } else {
            if (count < max) { out[count].s = s; out[count].start = seg; out[count].len = end - seg; }
            count++;
        }
    }
    for (int k = 0; k < count && k < max; k++)
        trim_span(&out[k]);
    return count;
}

/* ------------------------------------------------------------------ */
/* Device-code extraction                                             */
/* ------------------------------------------------------------------ */
typedef struct { size_t start, end; } range;

static int is_device_token(const char *s, size_t st, size_t len)
{
    return (len == 10 && strncmp(s + st, "__global__", 10) == 0) ||
           (len == 9  && strncmp(s + st, "__device__", 9) == 0);
}

static int extract_devices(const char *s, size_t n, range *ranges, int max)
{
    int count = 0;
    size_t i = 0;
    while (i < n) {
        size_t j = skip_noncode(s, n, i);
        if (j != i) { i = j; continue; }
        if (ident_start(s[i])) {
            size_t st = i;
            while (i < n && ident_char(s[i]))
                i++;
            if (!is_device_token(s, st, i - st))
                continue;
            size_t k = i;
            int done = 0;
            while (k < n && !done) {
                size_t m = skip_noncode(s, n, k);
                if (m != k) { k = m; continue; }
                if (s[k] == '{') {
                    size_t e = find_matching(s, n, k, '{', '}');
                    if (e == (size_t)-1) { k = n; break; }
                    if (count < max) ranges[count++] = (range){ st, e + 1 };
                    i = e + 1;
                    done = 1;
                } else if (s[k] == ';') {
                    if (count < max) ranges[count++] = (range){ st, k + 1 };
                    i = k + 1;
                    done = 1;
                } else {
                    k++;
                }
            }
            if (!done) i = n;
            continue;
        }
        i++;
    }
    return count;
}

/* ------------------------------------------------------------------ */
/* Emit helpers                                                       */
/* ------------------------------------------------------------------ */
static void emit_c_literal(buf *out, const char *src, size_t n)
{
    bputs(out, "static const char __mudac_kernel_source[] =\n\"");
    for (size_t i = 0; i < n; i++) {
        unsigned char c = (unsigned char)src[i];
        if (c == '\r')
            continue;
        if (c == '\n') { bputs(out, "\\n\"\n\""); continue; }
        if (c == '\\') { bputs(out, "\\\\"); continue; }
        if (c == '"')  { bputs(out, "\\\""); continue; }
        if (c == '?')  { bputs(out, "\\?");  continue; }
        if (c == '\t') { bputs(out, "\\t");  continue; }
        if (c < 0x20)  { bprintf(out, "\\x%02x", c); continue; }
        bputc(out, (char)c);
    }
    bputs(out, "\";\n\n");
}

#define SP_ARG(sp) (int)(sp)->len, (sp)->s + (sp)->start

static void emit_launch(buf *out, const char *s,
                        size_t name_off, size_t name_len,
                        span *cfg, int ncfg, span *args, int nargs)
{
    span one   = { "1", 0, 1 };
    span zero  = { "0", 0, 1 };
    span null  = { "NULL", 0, 4 };
    span grid  = ncfg > 0 ? cfg[0] : one;
    span block = ncfg > 1 ? cfg[1] : one;
    span shared = ncfg > 2 ? cfg[2] : zero;
    span stream = ncfg > 3 ? cfg[3] : null;

    bputs(out, "({ __mudac_init(); ");
    if (nargs > 0) {
        for (int i = 0; i < nargs; i++)
            bprintf(out, "MUDAC_AUTO __mudac_a%d = (%.*s); ", i, SP_ARG(&args[i]));
        bputs(out, "void* __mudac_args[] = {");
        for (int i = 0; i < nargs; i++)
            bprintf(out, "&__mudac_a%d,", i);
        bputs(out, "}; ");
    } else {
        bputs(out, "void* __mudac_args[] = {0}; ");
    }

    bprintf(out, "mudartLaunchByName(\"%.*s\", ",
            (int)name_len, s + name_off);
    bprintf(out, "MUDAC_DIMX(%.*s), MUDAC_DIMY(%.*s), MUDAC_DIMZ(%.*s), ",
            SP_ARG(&grid), SP_ARG(&grid), SP_ARG(&grid));
    bprintf(out, "MUDAC_DIMX(%.*s), MUDAC_DIMY(%.*s), MUDAC_DIMZ(%.*s), ",
            SP_ARG(&block), SP_ARG(&block), SP_ARG(&block));
    bprintf(out, "(size_t)(%.*s), (%.*s), %d, __mudac_args); })",
            SP_ARG(&shared), SP_ARG(&stream), nargs);
}

/* ------------------------------------------------------------------ */
/* Launch rewriting                                                   */
/* ------------------------------------------------------------------ */
static void rewrite_launches(const char *s, size_t n, buf *out)
{
    size_t i = 0;
    while (i < n) {
        size_t j = skip_noncode(s, n, i);
        if (j != i) {
            bput(out, s + i, j - i);
            i = j;
            continue;
        }
        if (ident_start(s[i])) {
            size_t st = i;
            while (i < n && ident_char(s[i]))
                i++;
            size_t name_len = i - st;
            size_t p = skip_ws(s, n, i);
            if (p + 2 < n && s[p] == '<' && s[p + 1] == '<' && s[p + 2] == '<') {
                size_t close = find_triple_gt(s, n, p + 3);
                size_t paren = close == (size_t)-1 ? (size_t)-1 : skip_ws(s, n, close + 3);
                if (close != (size_t)-1 && paren < n && s[paren] == '(') {
                    size_t q = find_matching(s, n, paren, '(', ')');
                    if (q != (size_t)-1) {
                        span cfg[4];
                        int ncfg = split_top(s, p + 3, close, cfg, 4);
                        span args[64];
                        int nargs = (q == paren + 1)
                                        ? 0
                                        : split_top(s, paren + 1, q, args, 64);
                        emit_launch(out, s, st, name_len, cfg, ncfg, args, nargs);
                        i = q + 1;
                        continue;
                    }
                }
            }
            bput(out, s + st, name_len);
            continue;
        }
        bputc(out, s[i]);
        i++;
    }
}

/* ------------------------------------------------------------------ */
/* Host preprocessing: drop CUDA includes                             */
/* ------------------------------------------------------------------ */
static void emit_host(buf *out, const char *s, size_t n)
{
    size_t i = 0;
    while (i < n) {
        size_t line_end = i;
        while (line_end < n && s[line_end] != '\n')
            line_end++;
        const char *line = s + i;
        size_t len = line_end - i;

        size_t k = 0;
        while (k < len && isspace((unsigned char)line[k]))
            k++;
        int is_include = (k < len && line[k] == '#');
        if (is_include) {
            /* look for the word "include" and a cuda token on this line */
            int has_include = 0, has_cuda = 0;
            for (size_t t = k; t + 7 <= len; t++) {
                if (strncmp(line + t, "include", 7) == 0 &&
                    (t == 0 || !ident_char(line[t - 1])) &&
                    (t + 7 == len || !ident_char(line[t + 7])))
                    has_include = 1;
            }
            for (size_t t = 0; t + 4 <= len; t++)
                if (strncmp(line + t, "cuda", 4) == 0)
                    has_cuda = 1;
            if (has_include && has_cuda) {
                bputc(out, '\n');
                i = line_end + (line_end < n ? 1 : 0);
                continue;
            }
        }
        bput(out, line, len);
        if (line_end < n)
            bputc(out, '\n');
        i = line_end + (line_end < n ? 1 : 0);
    }
}

/* ------------------------------------------------------------------ */
/* main                                                               */
/* ------------------------------------------------------------------ */
int main(int argc, char **argv)
{
    const char *inpath = NULL;
    const char *outpath = NULL;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            outpath = argv[++i];
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printf("usage: mudac [input.cu] [-o output.c]\n");
            return 0;
        } else {
            inpath = argv[i];
        }
    }

    FILE *in = inpath ? fopen(inpath, "rb") : stdin;
    if (!in) {
        fprintf(stderr, "mudac: cannot open %s\n", inpath);
        return 1;
    }
    fseek(in, 0, SEEK_END);
    long sz = ftell(in);
    if (sz < 0) { fclose(in); return 1; }
    fseek(in, 0, SEEK_SET);
    char *src = (char *)malloc((size_t)sz + 1);
    size_t got = fread(src, 1, (size_t)sz, in);
    src[got] = '\0';
    if (in != stdin)
        fclose(in);
    size_t n = got;

    /* 1. extract device functions */
    range ranges[256];
    int nranges = extract_devices(src, n, ranges, 256);

    /* 2. build the device source string: #define/#pragma lines + ranges */
    buf dev;
    binit(&dev);
    {
        size_t i = 0;
        while (i < n) {
            size_t e = i;
            while (e < n && src[e] != '\n')
                e++;
            size_t k = i;
            while (k < e && isspace((unsigned char)src[k]))
                k++;
            if (k < e && src[k] == '#') {
                const char *p = src + k + 1;
                while (p < src + e && isspace((unsigned char)*p))
                    p++;
                if ((size_t)(src + e - p) >= 6 &&
                    (strncmp(p, "define", 6) == 0 || strncmp(p, "pragma", 6) == 0)) {
                    bput(&dev, src + i, e - i);
                    bputc(&dev, '\n');
                }
            }
            i = e + (e < n ? 1 : 0);
        }
    }
    for (int r = 0; r < nranges; r++) {
        bput(&dev, src + ranges[r].start, ranges[r].end - ranges[r].start);
        bputc(&dev, '\n');
    }

    /* 3. blank out device ranges in a copy of the source */
    char *host = (char *)malloc(n + 1);
    memcpy(host, src, n);
    host[n] = '\0';
    for (int r = 0; r < nranges; r++)
        for (size_t i = ranges[r].start; i < ranges[r].end; i++)
            if (host[i] != '\n')
                host[i] = ' ';

    /* 4. rewrite launches on the blanked host source */
    buf host_out;
    binit(&host_out);
    rewrite_launches(host, n, &host_out);

    /* 5. assemble: include + kernel source + init + host code */
    buf out;
    binit(&out);
    bputs(&out, "/* Generated by mudac. Do not edit. */\n");
    bputs(&out, "#include \"mudart.h\"\n\n");
    emit_c_literal(&out, dev.data ? dev.data : "", dev.len);
    bputs(&out,
          "static int __mudac_initialized = 0;\n"
          "static void __mudac_init(void) __attribute__((unused));\n"
          "static void __mudac_init(void)\n"
          "{\n"
          "    if (!__mudac_initialized) {\n"
          "        mudartLoadKernelSource(__mudac_kernel_source);\n"
          "        __mudac_initialized = 1;\n"
          "    }\n"
          "}\n\n");
    emit_host(&out, host_out.data ? host_out.data : "", host_out.len);

    /* 6. write */
    FILE *o = outpath ? fopen(outpath, "wb") : stdout;
    if (!o) {
        fprintf(stderr, "mudac: cannot write %s\n", outpath);
        return 1;
    }
    fwrite(out.data, 1, out.len, o);
    if (o != stdout)
        fclose(o);

    free(src); free(host);
    free(dev.data); free(host_out.data); free(out.data);
    return 0;
}
