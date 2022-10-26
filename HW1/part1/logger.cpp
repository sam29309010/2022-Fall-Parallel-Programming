#include "logger.h"
#include "PPintrin.h"
#include <string>
#include <map>

void Logger::addLog(const char *instruction, __pp_mask mask, int N)
{
  Log newLog;
  strcpy(newLog.instruction, instruction);
  newLog.mask = 0;
  for (int i = 0; i < N; i++)
  {
    if (mask.value[i])
    {
      newLog.mask |= (((unsigned long long)1) << i);
      stats.utilized_lane++;
    }
  }
  stats.total_lane += N;
  stats.total_instructions += (N > 0);
  log.push_back(newLog);
}

void Logger::printStats()
{
  printf("****************** Printing Vector Unit Statistics *******************\n");
  printf("Vector Width:              %d\n", VECTOR_WIDTH);
  printf("Total Vector Instructions: %lld\n", stats.total_instructions);
  printf("Vector Utilization:        %.1f%%\n", (double)stats.utilized_lane / stats.total_lane * 100);
  printf("Utilized Vector Lanes:     %lld\n", stats.utilized_lane);
  printf("Total Vector Lanes:        %lld\n", stats.total_lane);
}

void Logger::printLog()
{
  // printf("***************** Printing Vector Unit Execution Log *****************\n");
  // printf(" Instruction | Vector Lane Occupancy ('*' for active, '_' for inactive)\n");
  // printf("------------- --------------------------------------------------------\n");
  // for (int i = 0; i < log.size(); i++)
  // {
  //   printf("%12s | ", log[i].instruction);
  //   for (int j = 0; j < VECTOR_WIDTH; j++)
  //   {
  //     if (log[i].mask & (((unsigned long long)1) << j))
  //     {
  //       printf("*");
  //     }
  //     else
  //     {
  //       printf("_");
  //     }
  //   }
  //   printf("\n");
  // }

  // breakdown of instrinsics insturctions
  string inst_name;
  struct inst_stats {
    int inst_count;
    int inst_mask;
    int inst_total;
  };
  map<string, inst_stats> counter;
  for (int i = 0; i < log.size(); i++){
    inst_name = log[i].instruction;
    if (!counter.count(inst_name)){
      counter[inst_name] = inst_stats{0,0,0};
    }
    counter[inst_name].inst_count++;
    for (int j=0; j< VECTOR_WIDTH; j++){
      if (log[i].mask & (((unsigned long long)1) << j)){
        counter[inst_name].inst_mask++;
      }
    }
    counter[inst_name].inst_total += VECTOR_WIDTH;
  }

  printf("**************** Breakdown of Intrinsics Instructions ****************\n");
  printf(" Instruction | Inst. Count | Ele. Mask | Ele. Total | Avg. Utilization\n");
  printf("------------- ------------- ----------- ------------ -----------------\n");
  inst_stats inst_info;
  char inst_chars[12+1];
  for (auto it=counter.begin(); it!=counter.end(); ++it){
    inst_name = it->first;
    inst_info = it->second;
    strcpy(inst_chars, inst_name.c_str());
    inst_chars[inst_name.size()] = '\0';
    printf("%12s | ", inst_chars);
    printf("%11d | %9d | %10d | %.4f \n",inst_info.inst_count, inst_info.inst_mask, inst_info.inst_total, (float)inst_info.inst_mask / inst_info.inst_total);
  }
}
