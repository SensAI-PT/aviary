#ifndef COLIBRI_CLUSTER_RPC_H
#define COLIBRI_CLUSTER_RPC_H
/* Cross-node expert RPC for Aviary Phase 2. See docs/cluster_protocol.md.
 *
 * When AVIARY_CLUSTER=1, moe() may call cluster_try_remote() before local load.
 * Placement table: JSON at AVIARY_PLACEMENT (written by aviary-agent).
 *
 * Peer table shape:
 *   {"node_id":"...","peers":{"uuid":"host:port",...},
 *    "experts":{"layer:eid":"uuid",...}}
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
typedef SOCKET cluster_sock_t;
#define CLUSTER_INVALID_SOCKET INVALID_SOCKET
#define cluster_close(s) closesocket(s)
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <time.h>
typedef int cluster_sock_t;
#define CLUSTER_INVALID_SOCKET (-1)
#define cluster_close(s) close(s)
#endif

static int g_cluster_enabled;
static int g_rpc_timeout_ms = 150;
static char g_cluster_self[64];
static char g_cluster_job_id[128];
static char g_placement_path[4096];

/* peer host:port strings keyed by node uuid (small cluster) */
#define CLUSTER_MAX_PEERS 16
static char g_peer_id[CLUSTER_MAX_PEERS][64];
static char g_peer_addr[CLUSTER_MAX_PEERS][128];
static int g_n_peers;

/* expert placement: layer*10000+eid -> peer index (-1 = local) */
#define CLUSTER_MAX_EXPERTS 8192
static int g_expert_peer[CLUSTER_MAX_EXPERTS]; /* indexed by layer*256+eid */

/* Aviary layer_caps: max residency tier (0=disk, 1=RAM, 2=VRAM) per layer */
static int g_layer_cap[256];

static void cluster_reset_layer_caps(void){
    for(int i = 0; i < 256; i++) g_layer_cap[i] = 2;
}

static int cluster_layer_max_tier(int layer){
    if(layer < 0 || layer >= 256) return 2;
    return g_layer_cap[layer];
}

static int cluster_expert_key(int layer, int eid){ return layer * 256 + eid; }

static void cluster_set_job_id(const char *job_id){
    if(!job_id || !job_id[0] || !strcmp(job_id, "-")){ g_cluster_job_id[0]=0; return; }
    snprintf(g_cluster_job_id, sizeof(g_cluster_job_id), "%s", job_id);
}

static void cluster_emit_trace(int layer, int eid, const char *kind, const char *peer, uint32_t rpc_us){
    if(!g_cluster_enabled && !g_cluster_job_id[0]) return;
    printf("TRACE %d %d %s %s %u\n", layer, eid, kind, peer && peer[0] ? peer : "-", rpc_us);
    fflush(stdout);
}

static void cluster_trim(char *s){
    size_t n = strlen(s);
    while(n && (s[n-1]=='\n' || s[n-1]=='\r' || s[n-1]==' ')) s[--n]=0;
}

static const char *cluster_json_str(const char *json, const char *key, char *buf, size_t bufsz){
    char pat[128]; snprintf(pat,sizeof(pat),"\"%s\"",key);
    const char *p = strstr(json, pat);
    if(!p) return NULL;
    p = strchr(p+strlen(pat), '"');
    if(!p) return NULL;
    p++;
    const char *end = strchr(p, '"');
    if(!end || (size_t)(end-p) >= bufsz) return NULL;
    memcpy(buf, p, (size_t)(end-p)); buf[end-p] = 0;
    return buf;
}

static void cluster_load_placement(void){
    for(int i = 0; i < CLUSTER_MAX_EXPERTS; i++) g_expert_peer[i] = -1;
    g_n_peers = 0;
    cluster_reset_layer_caps();
    if(!g_placement_path[0]) return;
    FILE *f = fopen(g_placement_path, "r");
    if(!f) return;
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    if(sz < 1 || sz > 65536){ fclose(f); return; }
    char *json = malloc((size_t)sz+1);
    if(!json){ fclose(f); return; }
    if(fread(json,1,(size_t)sz,f)!=(size_t)sz){ free(json); fclose(f); return; }
    json[sz]=0; fclose(f);
    cluster_json_str(json, "node_id", g_cluster_self, sizeof(g_cluster_self));
    /* parse peers object — naive scan for "uuid":"host:port" */
    const char *peers = strstr(json, "\"peers\"");
    if(peers){
        const char *p = strchr(peers, '{');
        if(p) for(p++; g_n_peers < CLUSTER_MAX_PEERS && *p && *p != '}'; ){
            while(*p==' '||*p==',') p++;
            if(*p!='"') break;
            p++;
            char id[64]={0}; int ii=0;
            while(*p && *p!='"' && ii<63) id[ii++]=*p++;
            if(*p!='"') break; p++;
            while(*p && *p!=':') p++;
            if(*p==':') p++;
            while(*p && *p!='"') p++;
            if(*p!='"') break; p++;
            char addr[128]={0}; ii=0;
            while(*p && *p!='"' && ii<127) addr[ii++]=*p++;
            snprintf(g_peer_id[g_n_peers], sizeof(g_peer_id[g_n_peers]), "%s", id);
            snprintf(g_peer_addr[g_n_peers], sizeof(g_peer_addr[g_n_peers]), "%s", addr);
            g_n_peers++;
            if(*p=='"') p++;
        }
    }
    const char *exp = strstr(json, "\"experts\"");
    if(exp){
        const char *p = strchr(exp, '{');
        if(p) for(p++; *p && *p != '}'; ){
            while(*p==' '||*p==',') p++;
            if(*p!='"') break;
            p++;
            char key[32]={0}; int ki=0;
            while(*p && *p!='"' && ki<31) key[ki++]=*p++;
            if(*p!='"') break; p++;
            while(*p && *p!=':') p++;
            if(*p==':') p++;
            while(*p && *p!='"') p++;
            if(*p!='"') break; p++;
            char target[64]={0}; ki=0;
            while(*p && *p!='"' && ki<63) target[ki++]=*p++;
            int layer=-1, eid=-1;
            if(sscanf(key, "%d:%d", &layer, &eid)==2 && layer>=0 && eid>=0){
                int pk = cluster_expert_key(layer, eid);
                if(pk >= 0 && pk < CLUSTER_MAX_EXPERTS){
                    int pi = -1;
                    for(int j=0;j<g_n_peers;j++) if(!strcmp(g_peer_id[j], target)){ pi=j; break; }
                    g_expert_peer[pk] = pi;
                }
            }
            if(*p=='"') p++;
        }
    }
    const char *caps = strstr(json, "\"layer_caps\"");
    if(caps){
        const char *p = strchr(caps, '{');
        if(p) for(p++; *p && *p != '}'; ){
            while(*p==' '||*p==',') p++;
            if(*p!='"') break;
            p++;
            char key[16]={0}; int ki=0;
            while(*p && *p!='"' && ki<15) key[ki++]=*p++;
            if(*p!='"') break; p++;
            while(*p && *p!=':') p++;
            if(*p==':') p++;
            while(*p && (*p==' '||*p=='\t')) p++;
            int cap = atoi(p);
            int layer = atoi(key);
            if(layer >= 0 && layer < 256) g_layer_cap[layer] = cap;
            while(*p && *p!=',' && *p!='}') p++;
        }
    }
    free(json);
}

static void cluster_init(void){
    static int done;
    if(done) return;
    done = 1;
    const char *e = getenv("AVIARY_CLUSTER");
    g_cluster_enabled = e && *e && e[0] != '0';
    const char *tm = getenv("AVIARY_RPC_TIMEOUT_MS");
    if(tm && *tm) g_rpc_timeout_ms = atoi(tm);
    const char *pp = getenv("AVIARY_PLACEMENT");
    if(pp) snprintf(g_placement_path, sizeof(g_placement_path), "%s", pp);
    if(g_cluster_enabled) cluster_load_placement();
#ifdef _WIN32
    if(g_cluster_enabled){ WSADATA wsa; WSAStartup(MAKEWORD(2,2), &wsa); }
#endif
}

static void cluster_reload(void){ cluster_load_placement(); }

static int cluster_lookup(int layer, int eid, char *host_out, int host_sz, int *port_out){
    if(!g_cluster_enabled) return 0;
    int pk = cluster_expert_key(layer, eid);
    if(pk < 0 || pk >= CLUSTER_MAX_EXPERTS) return 0;
    int pi = g_expert_peer[pk];
    if(pi < 0 || pi >= g_n_peers) return 0;
    const char *addr = g_peer_addr[pi];
    const char *colon = strrchr(addr, ':');
    if(!colon) return 0;
    size_t hlen = (size_t)(colon - addr);
    if(hlen >= (size_t)host_sz) return 0;
    memcpy(host_out, addr, hlen); host_out[hlen] = 0;
    *port_out = atoi(colon+1);
    return 1;
}

static cluster_sock_t cluster_connect(const char *host, int port){
    char port_s[16]; snprintf(port_s, sizeof(port_s), "%d", port);
    struct addrinfo hints={0}, *res=NULL;
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    if(getaddrinfo(host, port_s, &hints, &res) || !res) return CLUSTER_INVALID_SOCKET;
    cluster_sock_t s = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    if(s == CLUSTER_INVALID_SOCKET){ freeaddrinfo(res); return s; }
#ifndef _WIN32
    int flags = fcntl(s, F_GETFL, 0);
    if(flags >= 0) fcntl(s, F_SETFL, flags | O_NONBLOCK);
#endif
    int r = connect(s, res->ai_addr, (socklen_t)res->ai_addrlen);
    freeaddrinfo(res);
    if(r < 0
#ifdef _WIN32
       && WSAGetLastError() != WSAEWOULDBLOCK
#else
       && errno != EINPROGRESS
#endif
    ){ cluster_close(s); return CLUSTER_INVALID_SOCKET; }
    return s;
}

static int cluster_wait_writable(cluster_sock_t s, int ms){
    fd_set wfds; FD_ZERO(&wfds);
#ifndef _WIN32
    FD_SET(s, &wfds);
    struct timeval tv={ms/1000, (ms%1000)*1000};
    return select((int)s+1, NULL, &wfds, NULL, &tv) > 0;
#else
    FD_SET((SOCKET)s, &wfds);
    struct timeval tv={ms/1000, (ms%1000)*1000};
    return select(0, NULL, &wfds, NULL, &tv) > 0;
#endif
}

static int cluster_wait_readable(cluster_sock_t s, int ms){
    fd_set rfds; FD_ZERO(&rfds);
#ifndef _WIN32
    FD_SET(s, &rfds);
    struct timeval tv={ms/1000, (ms%1000)*1000};
    return select((int)s+1, &rfds, NULL, NULL, &tv) > 0;
#else
    FD_SET((SOCKET)s, &rfds);
    struct timeval tv={ms/1000, (ms%1000)*1000};
    return select(0, &rfds, NULL, NULL, &tv) > 0;
#endif
}

static int cluster_send_all(cluster_sock_t s, const char *buf, size_t n, int ms){
    size_t sent = 0;
    while(sent < n){
        if(!cluster_wait_writable(s, ms)) return -1;
#ifdef _WIN32
        int r = send(s, buf+sent, (int)(n-sent), 0);
#else
        ssize_t r = send(s, buf+sent, n-sent, 0);
#endif
        if(r <= 0) return -1;
        sent += (size_t)r;
    }
    return 0;
}

static int cluster_read_line(cluster_sock_t s, char *buf, size_t bufsz, int ms){
    size_t n = 0;
    while(n + 1 < bufsz){
        if(!cluster_wait_readable(s, ms)) return -1;
        char c;
#ifdef _WIN32
        int r = recv(s, &c, 1, 0);
#else
        ssize_t r = recv(s, &c, 1, 0);
#endif
        if(r <= 0) return -1;
        if(c == '\n'){ buf[n]=0; return (int)n; }
        buf[n++] = c;
    }
    return -1;
}

/* Returns 0 on success (out filled), -1 on miss/timeout (caller falls back local). */
static int cluster_rpc_expert(int layer, int eid, const float *x_in, int hidden,
                               float *out_row, uint32_t *rpc_us_out){
    cluster_init();
    if(!g_cluster_enabled) return -1;
    char host[128]; int port = 0;
    if(!cluster_lookup(layer, eid, host, sizeof(host), &port)) return -1;
    cluster_sock_t s = cluster_connect(host, port);
    if(s == CLUSTER_INVALID_SOCKET) return -1;
    static unsigned req_seq;
    unsigned req_id = ++req_seq;
    size_t bytes = (size_t)hidden * sizeof(float);
    char hdr[256];
    if(g_cluster_job_id[0])
        snprintf(hdr, sizeof(hdr), "EXEC_EXPERT %u %d %d %zu %s\n",
                 req_id, layer, eid, bytes, g_cluster_job_id);
    else
        snprintf(hdr, sizeof(hdr), "EXEC_EXPERT %u %d %d %zu\n", req_id, layer, eid, bytes);
    struct timespec t0, t1;
#ifdef _WIN32
    (void)t0; (void)t1;
#else
    clock_gettime(CLOCK_MONOTONIC, &t0);
#endif
    if(cluster_send_all(s, hdr, strlen(hdr), g_rpc_timeout_ms) < 0){ cluster_close(s); return -1; }
    if(cluster_send_all(s, (const char*)x_in, bytes, g_rpc_timeout_ms) < 0){ cluster_close(s); return -1; }
    if(cluster_send_all(s, "\n", 1, g_rpc_timeout_ms) < 0){ cluster_close(s); return -1; }
    char line[512];
    if(cluster_read_line(s, line, sizeof(line), g_rpc_timeout_ms) < 0){ cluster_close(s); return -1; }
    if(strncmp(line, "EXPERT_MISS", 11)==0){ cluster_close(s); return -1; }
    unsigned rid; size_t rb;
    if(sscanf(line, "EXPERT_RESULT %u %zu", &rid, &rb) != 2 || rb != bytes){
        cluster_close(s); return -1;
    }
    size_t got = 0;
    while(got < bytes){
        if(!cluster_wait_readable(s, g_rpc_timeout_ms)){ cluster_close(s); return -1; }
#ifdef _WIN32
        int r = recv(s, ((char*)out_row)+got, (int)(bytes-got), 0);
#else
        ssize_t r = recv(s, ((char*)out_row)+got, bytes-got, 0);
#endif
        if(r <= 0){ cluster_close(s); return -1; }
        got += (size_t)r;
    }
    char term;
    cluster_read_line(s, (char*)&term, 1, g_rpc_timeout_ms); /* consume trailing newline if any */
    cluster_close(s);
#ifndef _WIN32
    clock_gettime(CLOCK_MONOTONIC, &t1);
    uint32_t rpc_us = (uint32_t)((t1.tv_sec - t0.tv_sec) * 1000000u + (t1.tv_nsec - t0.tv_nsec) / 1000u);
#else
    uint32_t rpc_us = (uint32_t)(g_rpc_timeout_ms * 1000);
#endif
    if(rpc_us_out) *rpc_us_out = rpc_us;
    cluster_emit_trace(layer, eid, "remote", host, rpc_us);
    return 0;
}

#endif
