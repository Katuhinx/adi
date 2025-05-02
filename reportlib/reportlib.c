#include "reportlib.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>

static int file_exists(const char *filename) {
    struct stat buffer;
    return stat(filename, &buffer) == 0;
}

void report_result(const char *basename, const char *args_string, double exec_time_sec) {
    char filename[256];
    snprintf(filename, sizeof(filename), "%s_report.txt", basename);

    int file_existed = file_exists(filename);

    FILE *report = fopen(filename, "a");
    if (!report) {
        perror("Ошибка открытия отчётного файла");
        return;
    }

    if (!file_existed || ftell(report) == 0) {
        time_t now = time(NULL);
        char time_str[64];
        strftime(time_str, sizeof(time_str), "%Y-%m-%d %H:%M:%S", localtime(&now));
        fprintf(report, "# Report created at %s\n", time_str);
    }

    fprintf(report, "%s %.6f\n", args_string, exec_time_sec);
    fclose(report);
}