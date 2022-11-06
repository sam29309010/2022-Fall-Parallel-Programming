#include "bfs.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstddef>
#include <omp.h>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

#define ROOT_NODE_ID 0
#define NOT_VISITED_MARKER -1

// TBD free vertex_set memory

void vertex_set_clear(vertex_set *list)
{
    list->count = 0;
}

void vertex_set_init(vertex_set *list, int count)
{
    list->max_vertices = count;
    list->vertices = (int *)malloc(sizeof(int) * list->max_vertices);
    vertex_set_clear(list);
}

// Take one step of "top-down" BFS.  For each vertex on the frontier,
// follow all outgoing edges, and add all neighboring vertices to the
// new_frontier.
void top_down_step(
    Graph g,
    vertex_set *frontier,
    int *distances,
    int depth,
    int *edge_counter_ptr
    )
{
    int next_count = 0;
    *edge_counter_ptr = 0;
    for (int i = 0; i < frontier->count; ++i)
    {

        int node = frontier->vertices[i];

        int start_edge = g->outgoing_starts[node];
        int end_edge = (node == g->num_nodes - 1) ? g->num_edges : g->outgoing_starts[node + 1];
        *edge_counter_ptr += end_edge - start_edge;

        // attempt to add all neighbors to the new frontier
        for (int neighbor = start_edge; neighbor < end_edge; neighbor++)
        {
            int outgoing = g->outgoing_edges[neighbor];

            if (distances[outgoing] == NOT_VISITED_MARKER)
            {
                distances[outgoing]= depth;
            }
        }
    }

    for (int i = 0; i < g->num_nodes; ++i){
        if (distances[i] == depth){
            frontier->vertices[next_count++] = i;
        }
    }

    frontier->count = next_count;
}

// Implements top-down BFS.
//
// Result of execution is that, for each node in the graph, the
// distance to the root is stored in sol.distances.
void bfs_top_down(Graph graph, solution *sol)
{

    vertex_set list1;
    // vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    // vertex_set_init(&list2, graph->num_nodes);

    vertex_set *frontier = &list1;
    // vertex_set *next_frontier = &list2;

    int depth = 1;
    int edge_counter;
    // initialize all nodes to NOT_VISITED
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    while (frontier->count != 0)
    {

#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif

        top_down_step(graph, frontier, sol->distances, depth, &edge_counter);

#ifdef VERBOSE
        double end_time = CycleTimer::currentSeconds();
        printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif
        // swap pointers
        // vertex_set *tmp = frontier;
        // frontier = next_frontier;
        // next_frontier = tmp;
        depth++;
    }
}

void bottom_up_step(
    Graph g,
    vertex_set *frontier,
    int *distances,
    int depth,
    int *edge_counter_ptr
    )
{
    int next_count = 0;
    *edge_counter_ptr = 0;
    frontier->count = 0;
    for (int node = 0; node < g->num_nodes; ++node){
        if (distances[node] == NOT_VISITED_MARKER){
            int start_edge = g->incoming_starts[node];
            int end_edge = (node == g->num_nodes - 1) ? g->num_edges : g->incoming_starts[node + 1];
            *edge_counter_ptr += ((node == g->num_nodes - 1) ? g->num_edges : g->outgoing_starts[node + 1]) - g->outgoing_starts[node];
            for (int neighbor = start_edge; neighbor < end_edge; neighbor++)
            {
                int incoming = g->incoming_edges[neighbor];
                if (distances[incoming] == depth - 1)
                {
                    frontier->vertices[next_count++] = node;
                    distances[node]= depth;
                    break;
                }
            }
        }
    }
    frontier->count = next_count;
}


void bfs_bottom_up(Graph graph, solution *sol)
{
    // For PP students:
    //
    // You will need to implement the "bottom up" BFS here as
    // described in the handout.
    //
    // As a result of your code's execution, sol.distances should be
    // correctly populated for all nodes in the graph.
    //
    // As was done in the top-down case, you may wish to organize your
    // code by creating subroutine bottom_up_step() that is called in
    // each step of the BFS process.

    vertex_set list1;
    vertex_set_init(&list1, graph->num_nodes);

    vertex_set *frontier = &list1;

    int depth = 1;
    int edge_counter;
    // initialize all nodes to NOT_VISITED
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    while (frontier->count != 0)
    {

#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif

        bottom_up_step(graph, frontier, sol->distances, depth, &edge_counter);

#ifdef VERBOSE
        double end_time = CycleTimer::currentSeconds();
        printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif
        depth++;
    }
}

void bfs_hybrid(Graph graph, solution *sol)
{
    // For PP students:
    //
    // You will need to implement the "hybrid" BFS here as
    // described in the handout.
    vertex_set list1;
    vertex_set_init(&list1, graph->num_nodes);

    vertex_set *frontier = &list1;

    int depth = 1;
    int is_top_down = 1;
    int edge_counter, unexplored_edges = graph->num_edges;
    const int alpha = 14, beta = 24;
    // initialize all nodes to NOT_VISITED
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    while (frontier->count != 0)
    {
#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif
        if ((is_top_down) && (edge_counter > (float) unexplored_edges / alpha)){
            is_top_down = 0;
        }
        else if ((!is_top_down) && (frontier->count < (float) graph->num_nodes / beta)){
            is_top_down = 1;
        }

        if (is_top_down)
            top_down_step(graph, frontier, sol->distances, depth, &edge_counter);
        else
            bottom_up_step(graph, frontier, sol->distances, depth, &edge_counter);
        unexplored_edges -= edge_counter;

#ifdef VERBOSE
        double end_time = CycleTimer::currentSeconds();
        printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif
        depth++;
    }
}
