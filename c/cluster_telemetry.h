#ifndef COLIBRI_CLUSTER_TELEMETRY_H
#define COLIBRI_CLUSTER_TELEMETRY_H
/* Shared ECOST batch telemetry for cluster placement (see docs/serve_protocol.md).
 * Include after <stdio.h>; zero overhead when ct_enabled is 0. */

#include <stdio.h>
#include <stdint.h>

#define CT_BATCH_MAX 32

static int ct_enabled;
static int ct_n;
static int ct_layer[CT_BATCH_MAX], ct_eid[CT_BATCH_MAX], ct_tier[CT_BATCH_MAX];
static uint32_t ct_load_us[CT_BATCH_MAX], ct_exec_us[CT_BATCH_MAX];

static void ct_init(void){
    const char *e = getenv("AVIARY_CLUSTER");
    ct_enabled = e && *e && e[0] != '0';
    ct_n = 0;
}

static void ct_record(int layer, int eid, int tier, uint32_t load_us, uint32_t exec_us){
    if(!ct_enabled) return;
    if(ct_n >= CT_BATCH_MAX){
        ct_n = 0; /* drop batch on overflow — telemetry is advisory */
    }
    ct_layer[ct_n] = layer;
    ct_eid[ct_n] = eid;
    ct_tier[ct_n] = tier;
    ct_load_us[ct_n] = load_us;
    ct_exec_us[ct_n] = exec_us;
    ct_n++;
}

static void ct_flush(void){
    if(!ct_enabled || ct_n < 1) return;
    printf("ECOST %d", ct_n);
    for(int i = 0; i < ct_n; i++)
        printf(" %d %d %d %u %u", ct_layer[i], ct_eid[i], ct_tier[i],
               ct_load_us[i], ct_exec_us[i]);
    printf("\n");
    fflush(stdout);
    ct_n = 0;
}

#endif
