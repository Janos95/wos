
in vec3 position;
in float currentSolution;
in int currentNumSamples;
in float[MAX_STEPS] random;

layout(rgb32i, binding = 0)
uniform sampler2D nodes;

layout(rgb32i, binding = 0)
uniform sampler2D points;

layout(binding = 1)
uniform sampler2D colorMap;

layout(location = 0)
uniform int numPoints;

layout(location = 1)
uniform int numNodes;

#define INVALID_NODE -1
#define ROOT 0

struct ResultNN{
    float distanceSquared;
    int pointIndex;
};

struct Node{
    int leftChild;
    int rightChild;
    int pointIndex;
};

Node getNode(int i){
    return texture(nodes, vec2(float(i)/float(numNodes),0.0));
}

vec3 getPoint(int i){
    return texture(points, vec2(float(i)/float(numPoints),0.0));
}

ResultNN nearestNeighbor(vec3 q){
    ResultNN result;
    result.distanceSquared = 1./0.;  /* there is no predefined inf macro */

    Node currentNode = getNode(ROOT);
    for(int d = 0; d < TREE_DEPTH; ++d){
        if(currentNode == INVALID_NODE)
            break;
        vec3 p = getPoint(currentNode.pointIndex);
        float distSq = (p - q).dot();
        if(distSq < result.distanceSquared){
            result.distanceSquared = distSq;
            result.pointIndex = node.pointIndex;
        }
        int cd = d % 3;
        if(q[cd] < p[cd]){ /* q is closer to left child */
            currentNode = getNode(currentNode.leftChild);
        } else { /* q is closer to right child */
            currentNode = getNode(currentNode.rightChild);
        }
    }
    return result;
}

float walkOfSpheres(vec3 x0){
    float dist;
    int idx;
    for(int j = 0; j < MAX_STEPS; ++j)
        ResultNN result = tree.nearestNeighbor(x);
        dist = sqrt(result.distanceSquared);
        x = x + dist * normalize(vec3(distr(engine), distr(engine), distr(engine)));
        if(dist < eps){
            return getBoundaryData(result.pointIndex);
        }
    }
    return ;
}

void main(){
    ResultWoS result = walkOfSpheres(position);
    float newSolution = (float(currentNumTrials) * currentSolution + sum) / float(result.trialsNeeded + currentNumTrials);
}