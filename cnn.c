#include <stdio.h>
#include <stdbool.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <assert.h>
#include <math.h>
#include <pthread.h>
#include "mnist_extended.h"
#define LR 0.0005 //learning rate
#define beta1 0.9 //liée à adam
#define beta2 0.999 //liée à adam
#define eps 1e-8
#define nb_sep1 10 //un des main de ce code vise a classifier les images en trois catégories, celles à 'gauche' 
#define nb_sep2 36 //de nb_sep1 (chiffres), entre nb_sep1 et nb_sep2 (majuscules) et à 'droite' de nb_sep2 (minuscules)
pthread_mutex_t mut;

typedef enum
{
    input,
    output,
    max_pool,
    avg_pool,
    conv,
    flat,
    relu,
    fc
}type_de_couche;
typedef struct Tab 
{
    double**** tab;//de dimensions d1 x d2 x d3
    int d0;
    int d1;
    int d2;
    int d3;
}tab;
typedef struct Layer
{
    type_de_couche type;
    tab z;
    tab w;
    tab b;
    tab sdw;//RMS
    tab sdb;//RMS
    tab vdb;//MOMENTUM
    tab vdw;//MOMENTUM
    tab dropout;
    int adam_t;//ADAM
}layer;
typedef struct Cnn
{
    layer* step;
    int nb_step;
    int spmb;
    int adam_t;
}cnn;
typedef struct Call_feedforward_thread 
{
    cnn* net;
    int num_thread;
} call_feedforward_thread;
typedef struct Call_backprop_thread
{
    tab* error;
    int expected;
    double* acc;
    int* compteur_precision;
    cnn* net;
    int num_thread;

}call_backprop_thread;
/*---------------------------------------------------------------------*/
//énumeration des types de couches 
enum type_de_couche; 
//structure stockant un tableau 4d et les tailles de chaque dimension
struct Tab; 
//structure stockant les informations d'une couche du réseau
struct Layer; 
//structure stokant le réseau
struct Cnn; 
//structure servant à passer plusieurs paramètres dans une fonction appelée en multithread
struct Call_feedforward_thread;  
//structure servant à passer plusieurs paramètres dans une fonction appelée en multithread
struct Call_backprop_thread;  
//fonction qui initialise un reseau
cnn* init(int nb_step, type_de_couche* types,int* dim,int input_size,int* sizes_w,int spmb); 
//fonction qui effectue la propagation avant dans le réseau avec du multithreading
void* thread_feedforward(void* t); 
//fonction qui appelle thread_feedforward sur les différents threads
void feedforward_en_parallele(cnn* net); 
//fonction qui effectue la propagation arriète dans le réseau avec du multithreading
void* thread_backprop(void* call_v);  
//fonction qui appelle thread_backprop sur les différents threads
void backprop_en_parallele(cnn* net,tab* error, int* expected, double* acc,int* compteur_precision); 
//fonction qui update les poids et biais
void update(cnn* net,tab* error); 
//les trois fonctions suivantes servent a l'initialisation aléatoire des valeurs du réseau
double lotterie(double x);
double nu_conv(double k, double val); 
double nu_dense(double val); 
//les 4 fonctions suivantes servent à la création de tableau avec différentes valeurs initiales
tab init_tab_dim_4(int s0,int s1,int s2, int s3);
tab init_tab_dim_4_alea_conv(int s0,int s1,int s2, int s3);  
tab init_tab_dim_4_alea_dense(int s0,int s1,int s2, int s3); 
tab init_tab_dropout_dim_4(int s0,int s1,int s2, int s3); 
//les 9 fonctions suivantes sont décrites par le nom et peu intéressantes
double apply_relu(double x) ;  
double apply_relu_der(double x);  
double max_tab_2(double** tab, int i,int j); 
double avg_tab_2(double** tab, int i,int j); 
double apply_sigmoid(double x); 
bool is_in(int* tab, int taille, int n ); 
bool tirage_dropout(double dp); 
void print_matrix(int **matrix, int n); 
tab init_tab_null(); 
//sert a récuperer une image sous le formati mnist transformée en python
double*** charger_tableaux();  
//sert a sauvegarder un réseau
void save_model(cnn* net, char* filename) ;
//sert à charger un réseau
cnn* load_model(char* filename) ; 
//exemple de main pour un entrainement, ici pour le réseau qui classifie si
//une lettre est un chiffre une minuscule ou une majuscule
int main_entrainement();  
//renvoie le caractère associé au numéro asci
char asci(int asci_code); 
//effectue la propagation avant sans multithreading
void feedforward_sans_thread(cnn* net);  
//effectue la propagation arrière sans multithreading
void backprop_sans_thread(cnn* net,tab* error, int* expected, double* acc,int* compteur_precision);
//exemple du reseau final qui charge les différents réseau préentrainés et les utilise
int main_reconnaissance(); 



/*---------------------------------------------------------------------*/


cnn* init(int nb_step, type_de_couche* types,int* dim,int input_size,int* sizes_w,int spmb)
{
    assert(nb_step>0);
    cnn* net = malloc(sizeof(cnn));
    net->nb_step=nb_step;
    net->step = malloc(nb_step*sizeof(layer));
    net->spmb = spmb;
    net->adam_t=0;
    tab last_layer_z;
    for(int st = 0 ; st < nb_step ; st++)
    {
        layer l;
        l.type=types[st];
        if(st>0) 
        {
            last_layer_z = net->step[st-1].z;
        }
        switch(types[st])
        {
            case input:
            {
                l.z = init_tab_dim_4(spmb,dim[st],input_size,input_size);
                l.b = init_tab_null();
                l.w = init_tab_null();
                break;
            }
            case output:
            {
                l.z = init_tab_dim_4(spmb,last_layer_z.d1,last_layer_z.d2,last_layer_z.d3);
                l.b = init_tab_null();
                l.w = init_tab_null();
                break;
            }
            case max_pool:
            {
                l.z = init_tab_dim_4(spmb,last_layer_z.d1,last_layer_z.d2 / 2,last_layer_z.d3 / 2);
                l.b = init_tab_null();
                l.w = init_tab_null();
                break;
            }
            case avg_pool:
            {
                l.z = init_tab_dim_4(spmb,last_layer_z.d1,last_layer_z.d2 / 2,last_layer_z.d3 / 2);
                break;
            }
            case conv:
            {
                l.z = init_tab_dim_4(spmb,dim[st],last_layer_z.d2 - sizes_w[st] + 1,
                                        last_layer_z.d3 - sizes_w[st] + 1);
                //last_layer_z.d2 - sizes_w[st] + 1 provient de la diminution de la taille après une couche convolutive
                l.w = init_tab_dim_4_alea_conv(dim[st],dim[st-1],sizes_w[st],sizes_w[st]);
                l.b = init_tab_dim_4_alea_conv(1,1,1,dim[st]);
                l.dropout = init_tab_dropout_dim_4(spmb,dim[st],last_layer_z.d2 - sizes_w[st] + 1,
                                        last_layer_z.d3 - sizes_w[st] + 1);
                l.sdb = init_tab_dim_4(1,1,1,dim[st]);
                l.vdb = init_tab_dim_4(1,1,1,dim[st]);
                l.sdw = init_tab_dim_4(dim[st],dim[st-1],sizes_w[st],sizes_w[st]);
                l.vdw = init_tab_dim_4(dim[st],dim[st-1],sizes_w[st],sizes_w[st]);
                l.adam_t = 0;
                break;
            }
            case flat:
            {
                l.z = init_tab_dim_4(spmb,1,last_layer_z.d1*
                                    last_layer_z.d2*last_layer_z.d3,1);
                l.b = init_tab_null();
                l.w = init_tab_null();
                break;
            }
            case relu:
            {
                l.z = init_tab_dim_4(spmb,last_layer_z.d1,last_layer_z.d2,last_layer_z.d3);
                l.b = init_tab_null();
                l.w = init_tab_null();
                break;
            }
            case fc:
            {
                l.z = init_tab_dim_4(spmb,1,sizes_w[st],1);
                l.w = init_tab_dim_4_alea_dense(1,1,sizes_w[st],last_layer_z.d2);
                l.b = init_tab_dim_4_alea_dense(1,1,1,sizes_w[st]);
                l.dropout = init_tab_dropout_dim_4(spmb,1,sizes_w[st],1);
                l.sdw = init_tab_dim_4(1,1,sizes_w[st],last_layer_z.d2);
                l.vdw = init_tab_dim_4(1,1,sizes_w[st],last_layer_z.d2);
                l.sdb = init_tab_dim_4(1,1,1,sizes_w[st]);
                l.vdb = init_tab_dim_4(1,1,1,sizes_w[st]);
                l.adam_t = 0;
                break;
            }
        }
        net->step[st] = l;
    }
    return net;
}

/*---------------------------------------------------------------------*/


void* thread_feedforward(void* t)
{
    /*
    on doit mettre un void, et donc caster en thread* au début de la fonction, car les appels à des fonctions gérées
    par des threads doivent être faits sur des fonctions de paramètre void (condition requise par la bibliothèque posix)
    */
    call_feedforward_thread* th = (call_feedforward_thread*) t;
    cnn* net = th->net;
    int a = th->num_thread;
    for(int st = 0 ; st < net->nb_step ; st++)
    {
        layer* l = &net->step[st];
        switch(l->type)
        {
            case input:
            {
                break;
            }
            case output:
            {
                for(int i = 0 ; i < net->step[st-1].z.d1 ; i++)
                {
                    double max = net->step[st-1].z.tab[a][i][0][0];
                    for(int h = 1 ; h < net->step[st-1].z.d2 ; h++)
                    {
                        if(net->step[st-1].z.tab[a][i][h][0]>max)max=net->step[st-1].z.tab[a][i][h][0];
                    }
                    //on calcule la somme des exp pour le softmax :
                    double c = 0;
                    for(int j = 0 ; j < net->step[st-1].z.d2 ; j++) 
                    {
                        c+=exp(net->step[st-1].z.tab[a][i][j][0] - max);
                    }
                    assert(!isnan(c));
                    for(int j = 0 ; j < net->step[st-1].z.d2 ; j++)
                    {
                        l->z.tab[a][i][j][0] = exp((net->step[st-1].z.tab[a][i][j][0]) - max)/c;
                    }
                }
                break;
            }
            case max_pool:
            {
                for(int i = 0 ; i < l->z.d1 ; i++)
                {
                    for(int j = 0 ; j < l->z.d2 ; j++)
                    {
                        for(int k = 0 ; k < l->z.d3 ; k++)
                        {
                            l->z.tab[a][i][j][k]=max_tab_2(net->step[st-1].z.tab[a][i],j*2,k*2);
                        }
                    }
                }
                break;
            }
            case avg_pool:
            {
                for(int i = 0 ; i < l->z.d1 ; i++)
                {
                    for(int j = 0 ; j < l->z.d2 ; j++)
                    {
                        for(int k = 0 ; k < l->z.d3 ; k++)
                        {
                            l->z.tab[a][i][j][k]=avg_tab_2(net->step[st-1].z.tab[a][i],j*2,k*2);
                        }
                    }
                }
                break;
            }
            case conv:
            {
                for(int i = 0 ; i < l->w.d0 ; i++)
                {
                    for(int j = 0 ; j < l->z.d2 ; j++)
                    {
                        for(int k = 0 ; k < l->z.d3; k++)
                        {
                            double c = 0;
                            for(int i1 = 0 ; i1 < l->w.d1 ; i1++)//on somme sur toute les dimensions
                            {
                                for(int j1 = 0 ; j1 < l->w.d2 ; j1++)
                                {
                                    for(int k1 = 0 ; k1 < l->w.d2 ; k1++)
                                    {
                                        c+=l->w.tab[i][i1][j1][k1]*net->step[st-1].z.tab[a][i1][j+j1][k+k1];
                                    }
                                }
                            c+=l->b.tab[0][0][0][i];
                            l->z.tab[a][i][j][k] = c;
                            }
                        }
                    }
                    
                }
                break;
            }
            case flat:
            {
                for(int i = 0 ; i < net->step[st-1].z.d1 ; i++)
                {
                    for(int j = 0 ; j < net->step[st-1].z.d2 ; j++)
                    {
                        for(int k = 0 ; k < net->step[st-1].z.d3 ; k++)
                        {
                            l->z.tab[a][0][i*net->step[st-1].z.d2*net->step[st-1].z.d3+
                                            j*net->step[st-1].z.d3+k][0] 
                            = 
                            net->step[st-1].z.tab[a][i][j][k];
                        }
                    }
                }
                break;
            }
            case relu:
            {
                for(int i = 0 ; i < net->step[st-1].z.d1 ; i++)
                {
                    for(int j = 0 ; j < net->step[st-1].z.d2 ; j++)
                    {
                        for(int k = 0 ; k < net->step[st-1].z.d3 ; k++)
                        {
                            l->z.tab[a][i][j][k] = apply_relu(net->step[st-1].z.tab[a][i][j][k]);
                        }
                    }
                }
                break;
            }
            case fc:
            {
                for(int i = 0 ; i < l->z.d2 ; i++)
                {
                    double c = 0;
                    for(int j = 0 ; j < net->step[st-1].z.d2 ; j++)
                    {
                        c+=l->w.tab[0][0][i][j]*net->step[st-1].z.tab[a][0][j][0];
                    }
                    c+=l->b.tab[0][0][0][i];
                    l->z.tab[a][0][i][0] = c;
                }
                break;
            }
        }
    }
    return NULL;
}

/*---------------------------------------------------------------------*/


void feedforward_en_parallele(cnn* net)
{
    call_feedforward_thread* th1 = malloc(sizeof(call_feedforward_thread));
    call_feedforward_thread* th2 = malloc(sizeof(call_feedforward_thread));
    call_feedforward_thread* th3 = malloc(sizeof(call_feedforward_thread));
    call_feedforward_thread* th4 = malloc(sizeof(call_feedforward_thread));
    call_feedforward_thread* th5 = malloc(sizeof(call_feedforward_thread));
    call_feedforward_thread* th6 = malloc(sizeof(call_feedforward_thread));
    call_feedforward_thread* th7 = malloc(sizeof(call_feedforward_thread));
    call_feedforward_thread* th8 = malloc(sizeof(call_feedforward_thread));


    th1->net=net;
    th2->net=net;
    th3->net=net;
    th4->net=net;
    th5->net=net;
    th6->net=net;
    th7->net=net;
    th8->net=net;

    th1->num_thread=0;
    th2->num_thread=1;
    th3->num_thread=2;
    th4->num_thread=3;
    th5->num_thread=4;
    th6->num_thread=5;
    th7->num_thread=6;
    th8->num_thread=7;

    pthread_t t1,t2,t3,t4,t5,t6,t7,t8;

    pthread_create(&t1,NULL,thread_feedforward,th1);
    pthread_create(&t2,NULL,thread_feedforward,th2);
    pthread_create(&t3,NULL,thread_feedforward,th3);
    pthread_create(&t4,NULL,thread_feedforward,th4);
    pthread_create(&t5,NULL,thread_feedforward,th5);
    pthread_create(&t6,NULL,thread_feedforward,th6);
    pthread_create(&t7,NULL,thread_feedforward,th7);
    pthread_create(&t8,NULL,thread_feedforward,th8);

    pthread_join(t1, NULL);
    pthread_join(t2, NULL);
    pthread_join(t3, NULL);
    pthread_join(t4, NULL);
    pthread_join(t5, NULL);
    pthread_join(t6, NULL);
    pthread_join(t7, NULL);
    pthread_join(t8, NULL);

    free(th1);
    free(th2);
    free(th3);
    free(th4);
    free(th5);
    free(th6);
    free(th7);
    free(th8);
}

/*---------------------------------------------------------------------*/


void* thread_backprop(void* call_v)
{
    //on met en place les paramètres de la fonction
    call_backprop_thread*  call = (call_backprop_thread*)call_v;
    cnn* net = call->net;
    int a = call->num_thread;
    tab* error = call->error;
    int expected = call-> expected;
    double* acc = call->acc;
    int* compteur_precision = call->compteur_precision;

    int i_max = 0;
    double max = -INFINITY;
    for(int i = 0 ; i < net->step[net->nb_step - 1].z.d2 ; i++)
    {
        if(net->step[net->nb_step - 1].z.tab[a][0][i][0] > max)//on cherche l'indice du maximum
        {
            max=net->step[net->nb_step - 1].z.tab[a][0][i][0];
            i_max=i;
        }
        if(i == expected)
        {
            error[net->nb_step - 1].tab[a][0][i][0] = net->step[net->nb_step - 1].z.tab[a][0][i][0]-1;
            pthread_mutex_lock(&mut);
            (*acc)-=log(net->step[net->nb_step - 1].z.tab[a][0][i][0]+eps);//acc est une variable partagée je lock un mutex pour éviter des problèmes
            pthread_mutex_unlock(&mut);
        } 
        else
        {
            error[net->nb_step - 1].tab[a][0][i][0] = net->step[net->nb_step - 1].z.tab[a][0][i][0];  
        }
    }  
    if(i_max==expected)
    {
        pthread_mutex_lock(&mut);
        (*compteur_precision)++;//c'est aussi une variale partagée
        pthread_mutex_unlock(&mut);
    }
    for(int st = net->nb_step - 2 ; st > -1 ; st--)
    {
        tab* p = &error[st+1];
        tab* l = &error[st];
        switch(net->step[st+1].type)//on agit selon les types de couche
        {
            case input:
            {
                assert(1==1);
                break;
            }
            case output:
            {
                for(int a = 0 ; a < net->spmb; a++)
                {
                    for(int i = 0 ; i < l->d1 ; i++)
                    {
                        for(int j = 0 ; j < l->d2 ; j++)
                        {
                            for(int k = 0 ; k < l->d3 ; k++)
                            {
                                l->tab[a][i][j][k] = p->tab[a][i][j][k];
                                
                            }
                        }
                    }
                }
                break;
            }
            case max_pool:
            {
                for(int a = 0 ; a < net->spmb; a++)
                {
                    for(int i = 0 ; i < l->d1 ; i++)
                    {
                        for(int j = 0 ; j < l->d2 ; j++)
                        {
                            for(int k = 0 ; k < l->d3 ; k++)
                            {
                                if(k/2 < net->step[st+1].z.d3 && j/2 < net->step[st+1].z.d2)
                                {
                                    if(net->step[st].z.tab[a][i][j][k] == net->step[st+1].z.tab[a][i][j/2][k/2])//si c'est le max qui a été choisi par le pool
                                    {
                                        l->tab[a][i][j][k] = p->tab[a][i][j/2][k/2];
                                    }
                                    else 
                                    {
                                        l->tab[a][i][j][k] = 0;
                                    }
                                }
                            }
                        }
                    }
                }
                break;
            }
            case avg_pool:
            {
                for(int a = 0 ; a < l->d0; a++)
                {
                    for(int i = 0 ; i < l->d1 ; i++)
                    {
                        for(int j = 0 ; j < l->d2 ; j++)
                        {
                            for(int k = 0 ; k < l->d3 ; k++)
                            {
                                l->tab[a][i][j][k] = p->tab[a][i][j/2][k/2];
                            }
                        }
                    }
                }
                break;
            }
            case conv:
            {   
                for(int a = 0 ; a < net->spmb ; a++)
                {
                    for(int i = 0 ; i < net->step[st].z.d1 ; i++)
                    {
                        for(int j = 0 ; j < net->step[st].z.d2 ; j++)
                        {
                            for(int k = 0 ; k < net->step[st].z.d3 ; k++)
                            {
                                double c = 0;
                                for(int i1 = 0 ; i1 < net->step[st+1].z.d1 ; i1++)
                                {
                                    for(int j1 = 0 ; j1 < net->step[st+1].w.d2 ; j1++)
                                    {
                                        for(int k1 = 0 ; k1 < net->step[st+1].w.d2 ; k1++)
                                        {
                                            if(j-j1 > -1 && j-j1 < net->step[st+1].z.d2 && k-k1 > -1 && k-k1 < net->step[st+1].z.d3)
                                            //pour éviter tout dépassement d'indice
                                                {
                                                    c+=p->tab[a][i1][j-j1][k-k1]*
                                                        net->step[st+1].w.tab[i1][i][j1][k1];
                                                }
                                        }
                                    }
                                }
                                l->tab[a][i][j][k] = c;
                            }
                        }
                    }
                }
                break;
            }
            case flat:
            {
                for(int a = 0 ; a < net->spmb; a++)
                {
                    for(int i = 0 ; i < l->d1 ; i++)
                    {
                        for(int j = 0 ; j < l->d2 ; j++)
                        {
                            for(int k = 0 ; k < l->d3 ; k++)
                            {
                                l->tab[a][i][j][k] = p->tab[a][0][i*l->d2*l->d3+j*l->d3+k][0];
                            }
                        }
                    }
                }
                break;
            }
            case relu:
            {
                for(int a = 0 ; a < net->spmb; a++)
                {
                    for(int i = 0 ; i < l->d1 ; i++)
                    {
                        for(int j = 0 ; j < l->d2 ; j++)
                        {
                            for(int k = 0 ; k < l->d3 ; k++)
                            {
                                l->tab[a][i][j][k] =p->tab[a][i][j][k]* apply_relu_der(net->step[st].z.tab[a][i][j][k]);
                            }
                        }
                    }
                }
                break;
            }
            case fc:
            {
                for(int a = 0 ; a < net->spmb ; a++)
                {
                    
                    for(int i = 0 ; i < net->step[st].z.d2 ; i++)
                    {
                        double c = 0;
                        for(int j = 0 ; j < net->step[st+1].z.d2 ; j++)
                        {
                            c+=net->step[st+1].w.tab[0][0][j][i]*
                                p->tab[a][0][j][0];
                        }
                        l->tab[a][0][i][0] = c;
                        
                    }
                }
                break;
            }
        }
    }
    return NULL;
}

/*---------------------------------------------------------------------*/


void backprop_en_parallele(cnn* net,tab* error, int* expected, double* acc,int* compteur_precision)
{
    call_backprop_thread* th1 = malloc(sizeof(call_backprop_thread));
    call_backprop_thread* th2 = malloc(sizeof(call_backprop_thread));
    call_backprop_thread* th3 = malloc(sizeof(call_backprop_thread));
    call_backprop_thread* th4 = malloc(sizeof(call_backprop_thread));
    call_backprop_thread* th5 = malloc(sizeof(call_backprop_thread));
    call_backprop_thread* th6 = malloc(sizeof(call_backprop_thread));
    call_backprop_thread* th7 = malloc(sizeof(call_backprop_thread));
    call_backprop_thread* th8 = malloc(sizeof(call_backprop_thread));

    th1->net=net;
    th2->net=net;
    th3->net=net;
    th4->net=net;
    th5->net=net;
    th6->net=net;
    th7->net=net;
    th8->net=net;

    th1->num_thread=0;
    th2->num_thread=1;
    th3->num_thread=2;
    th4->num_thread=3;
    th5->num_thread=4;
    th6->num_thread=5;
    th7->num_thread=6;
    th8->num_thread=7;

    th1->acc=acc;
    th2->acc=acc;
    th3->acc=acc;
    th4->acc=acc;
    th5->acc=acc;
    th6->acc=acc;
    th7->acc=acc;
    th8->acc=acc;

    th1->error=error;
    th2->error=error;
    th3->error=error;
    th4->error=error;
    th5->error=error;
    th6->error=error;
    th7->error=error;
    th8->error=error;
    
    th1->expected = expected[0];
    th2->expected = expected[1];
    th3->expected = expected[2];
    th4->expected = expected[3];
    th5->expected = expected[4];
    th6->expected = expected[5];
    th7->expected = expected[6];
    th8->expected = expected[7];

    th1->compteur_precision=compteur_precision;
    th2->compteur_precision=compteur_precision;
    th3->compteur_precision=compteur_precision;
    th4->compteur_precision=compteur_precision;
    th5->compteur_precision=compteur_precision;
    th6->compteur_precision=compteur_precision;
    th7->compteur_precision=compteur_precision;
    th8->compteur_precision=compteur_precision;

    pthread_t t1,t2,t3,t4,t5,t6,t7,t8;

    pthread_create(&t1,NULL,thread_backprop,th1);
    pthread_create(&t2,NULL,thread_backprop,th2);
    pthread_create(&t3,NULL,thread_backprop,th3);
    pthread_create(&t4,NULL,thread_backprop,th4);
    pthread_create(&t5,NULL,thread_backprop,th5);
    pthread_create(&t6,NULL,thread_backprop,th6);
    pthread_create(&t7,NULL,thread_backprop,th7);
    pthread_create(&t8,NULL,thread_backprop,th8);

    pthread_join(t1, NULL);
    pthread_join(t2, NULL);
    pthread_join(t3, NULL);
    pthread_join(t4, NULL);
    pthread_join(t5, NULL);
    pthread_join(t6, NULL);
    pthread_join(t7, NULL);
    pthread_join(t8, NULL);

    free(th1);
    free(th2);
    free(th3);
    free(th4);
    free(th5);
    free(th6);
    free(th7);
    free(th8);
}

/*---------------------------------------------------------------------*/


void update(cnn* net,tab* error)
{
    //modification des paramètres
    double last;
    net->adam_t++;
    double beta_t1 = pow(beta1,net->adam_t);
    double beta_t2=pow(beta2,net->adam_t);
    for(int st = 0 ; st < net->nb_step ; st++)
    {
        layer* lay = &net->step[st];
        //assez indigeste, c'est une mise en oeuvre du résultat donné par le calcul de dérivée partiel et de adam,
        //s'y référer pour comprendre ce qu'il se passe :)
        if(net->step[st].type == conv)
        {
            //mise a jour des poids de la couche convolutive
            for(int i = 0 ; i < net->step[st].w.d0 ; i++)
            {
                for(int j = 0 ; j < net->step[st].w.d1 ; j++)
                {
                    for(int k = 0 ; k < net->step[st].w.d2 ; k++)
                    {
                        for(int l = 0 ; l < net->step[st].w.d2 ; l++)
                        {
                            double c = 0;
                            for(int a = 0 ; a < net->spmb ; a++)
                            {
                                for(int k1 = 0 ; k1 < error[st].d2 ; k1++)
                                {
                                    for(int l1 = 0 ; l1 < error[st].d3 ; l1++)
                                    {
                                        if(k+k1 < net->step[st-1].z.d2 &&
                                            l+l1 < net->step[st-1].z.d3)
                                        {
                                            c+=net->step[st-1].z.tab[a][j][k+k1][l+l1]*
                                                error[st].tab[a][i][k1][l1];
                                        }
                                    }
                                }
                            }
                            double cp = c/net->spmb;
                            last = lay->sdw.tab[i][j][k][l];
                            lay->sdw.tab[i][j][k][l] = beta1*last+(1-beta1)*cp;
                            double m = lay->sdw.tab[i][j][k][l]/((1-beta_t1) );

                            last = lay->vdw.tab[i][j][k][l];
                            lay->vdw.tab[i][j][k][l] = beta2*last+(1-beta2)*cp*cp;
                            double v =  lay->vdw.tab[i][j][k][l]/(1-beta_t2);
                            lay->w.tab[i][j][k][l]-=LR*m/(sqrt(v+eps));
                        }
                    }
                }
            }
            //mise a jour des biais de la couche convolutive
            for(int i = 0 ; i < net->step[st].b.d3 ; i++)
            {
                double c = 0;
                for(int a = 0 ; a < net->spmb ; a++)
                {
                    for(int j = 0 ; j < error[st].d2 ; j++)
                    {
                        for(int k = 0 ; k < error[st].d3 ; k++)
                        {
                            c+=error[st].tab[a][i][j][k];
                        }
                    }
                }
                double cp = c/net->spmb;
                last = lay->sdb.tab[0][0][0][i];
                lay->sdb.tab[0][0][0][i] = beta1*last+(1-beta1)*cp;
                double m = lay->sdb.tab[0][0][0][i]/(1-beta_t1);

                last = lay->vdb.tab[0][0][0][i];
                lay->vdb.tab[0][0][0][i] = beta2*last+(1-beta2)*cp*cp;
                double v = lay->vdb.tab[0][0][0][i]/(1-beta_t2);
                lay->b.tab[0][0][0][i] -= LR*m/sqrt(v+eps);
            }
        }
        else if (net->step[st].type == fc)
        {
            //mise a jour des poids de la fully connected layer
            for(int i = 0 ; i < net->step[st].w.d2 ; i++)
            {
                for(int j = 0 ; j < net->step[st].w.d3 ; j++)
                {
                    double c = 0;
                    for(int a = 0 ; a < net->spmb ; a++)
                    {
                        c+=error[st].tab[a][0][i][0]*
                            net->step[st-1].z.tab[a][0][j][0];
                    }
                    double cp = c/net->spmb; 
                    last = lay->sdw.tab[0][0][i][j];
                    lay->sdw.tab[0][0][i][j] = beta1*last+(1-beta1)*cp;
                    double m = lay->sdw.tab[0][0][i][j]/((1-beta_t1));

                    last = lay->vdw.tab[0][0][i][j];
                    lay->vdw.tab[0][0][i][j] = beta2*last+(1-beta2)*cp*cp;
                    double v =  lay->vdw.tab[0][0][i][j]/(1-beta_t2);
                    lay->w.tab[0][0][i][j]-=LR*m/(sqrt(v+eps));
                }
            }
            //mise a jour des biais de la fully connected layer
            for(int i = 0 ; i < net->step[st].b.d3 ; i++)
            {
                double c = 0;
                for(int a = 0 ; a < net->spmb ; a++)
                {
                    c+=error[st].tab[a][0][i][0];
                }
                double cp = c/net->spmb;
                last = lay->sdb.tab[0][0][0][i];
                lay->sdb.tab[0][0][0][i] = beta1*last+(1-beta1)*cp;
                double m = lay->sdb.tab[0][0][0][i]/(1-beta_t1);

                last = lay->vdb.tab[0][0][0][i];
                lay->vdb.tab[0][0][0][i] = beta2*last+(1-beta2)*cp*cp;
                double v = lay->vdb.tab[0][0][0][i]/(1-beta_t2);
                lay->b.tab[0][0][0][i] -= LR*m/sqrt(v+eps); 
            }
        }
    }
    //reinitialisation du tableau erreur
    for(int st = 0 ; st < net->nb_step ; st++)
    {
        for(int i = 0 ; i < error[st].d0 ; i++)
        {
            for(int j = 0 ; j < error[st].d1 ; j++)
            {
                for(int k = 0 ; k < error[st].d2 ; k++)
                {
                    for(int l = 0 ; l < error[st].d3 ; l++)
                    {
                        error[st].tab[i][j][k][l] = 0;
                    }
                }
            }
        }
    }
}

/*---------------------------------------------------------------------*/


double lotterie(double x)
{
  return x * (2 * (((double)rand()) / RAND_MAX) - 1);
}

/*---------------------------------------------------------------------*/


double nu_conv(double k, double val) 
{
  assert(k * k * val!=0);
  return (double)sqrtf(2. / (k * k * val));
}

/*---------------------------------------------------------------------*/


double nu_dense(double val)
{
    assert(val!=0);
    return (double)sqrtf(2. / val); 
}

/*---------------------------------------------------------------------*/


tab init_tab_dim_4(int s0,int s1,int s2, int s3)
{
    double**** t = malloc(s0*sizeof(double***));
    for(int a = 0 ; a < s0 ; a++)
    {
        t[a] = malloc(s1*sizeof(double**));
        for(int i = 0 ; i < s1 ; i++)
        {
            t[a][i] = malloc(s2*sizeof(double*));
            for(int j = 0 ; j < s2 ; j++)
            {
                t[a][i][j] = malloc(s3*sizeof(double));
                for(int k = 0 ; k < s3 ; k++)
                {
                    t[a][i][j][k] = (double)0;
                }
            }
        }
    }
    tab ta;
    ta.tab=t;
    ta.d0=s0;
    ta.d1=s1;
    ta.d2=s2;
    ta.d3=s3;
    return ta;
}

/*---------------------------------------------------------------------*/


tab init_tab_dim_4_alea_conv(int s0,int s1,int s2, int s3)
{
    double**** t = malloc(s0*sizeof(double***));
    for(int a = 0 ; a < s0 ; a++)
    {
        t[a] = malloc(s1*sizeof(double**));
        for(int i = 0 ; i < s1 ; i++)
        {
            t[a][i] = malloc(s2*sizeof(double*));
            for(int j = 0 ; j < s2 ; j++)
            {
                t[a][i][j] = malloc(s3*sizeof(double));
                for(int k = 0 ; k < s3 ; k++)
                {
                    double test = lotterie(nu_conv(s3,s0));
                    t[a][i][j][k] = test;
                }
            }
        }
    }
    tab ta;
    ta.tab=t;
    ta.d0=s0;
    ta.d1=s1;
    ta.d2=s2;
    ta.d3=s3;
    return ta;
}

/*---------------------------------------------------------------------*/


tab init_tab_dim_4_alea_dense(int s0,int s1,int s2, int s3)
{
    double**** t = malloc(s0*sizeof(double***));
    for(int a = 0 ; a < s0 ; a++)
    {
        t[a] = malloc(s1*sizeof(double**));
        for(int i = 0 ; i < s1 ; i++)
        {
            t[a][i] = malloc(s2*sizeof(double*));
            for(int j = 0 ; j < s2 ; j++)
            {
                t[a][i][j] = malloc(s3*sizeof(double));
                for(int k = 0 ; k < s3 ; k++)
                {
                    double test = lotterie(nu_dense(s3));
                    t[a][i][j][k] = test;
                }
            }
        }
    }
    tab ta;
    ta.tab=t;
    ta.d0=s0;
    ta.d1=s1;
    ta.d2=s2;
    ta.d3=s3;
    return ta;
}

/*---------------------------------------------------------------------*/


tab init_tab_dropout_dim_4(int s0,int s1,int s2, int s3)

{
    double**** t = malloc(s0*sizeof(double***));
    for(int a = 0 ; a < s0 ; a++)
    {
        t[a] = malloc(s1*sizeof(double**));
        for(int i = 0 ; i < s1 ; i++)
        {
            t[a][i] = malloc(s2*sizeof(double*));
            for(int j = 0 ; j < s2 ; j++)
            {
                t[a][i][j] = malloc(s3*sizeof(double));
                for(int k = 0 ; k < s3 ; k++)
                {
                    t[a][i][j][k] = (double)1;
                }
            }
        }
    }
    tab ta;
    ta.tab=t;
    ta.d0=s0;
    ta.d1=s1;
    ta.d2=s2;
    ta.d3=s3;
    return ta;
}

/*---------------------------------------------------------------------*/


double apply_relu(double x) {return (x>0 ? x:0) ;}


/*---------------------------------------------------------------------*/


double apply_relu_der(double x) {return (x>0 ? 1:0.01);}

/*---------------------------------------------------------------------*/


double max_tab_2(double** tab, int i,int j)
{
    double max = tab[i][j];
    for(int a = 0 ; a < 2 ; a++)
    {
        for(int b = 0 ; b < 2 ; b++)
        {
            if(tab[a][b] > max) max = tab[a][b];
        }
    }
    return max;
}

/*---------------------------------------------------------------------*/


double avg_tab_2(double** tab, int i,int j)
{
    double c=0;;
    for(int a = 0 ; a < 2 ; a++)
    {
        for(int b = 0 ; b < 2 ; b++)
        {
            c+=tab[a][b];
        }
    }
    return c/4;
}

/*---------------------------------------------------------------------*/


double apply_sigmoid(double x)
{
    return (double)1.f/(1.f+(float)exp(-x));
}

/*---------------------------------------------------------------------*/


bool is_in(int* tab, int taille, int n )
{
    for(int i = 0 ; i < taille ; i++) if(tab[i]==n) return true;
    return false;
}

/*---------------------------------------------------------------------*/


bool tirage_dropout(double dp) 
{
    double r = ((double) rand()) / RAND_MAX;
    return r >= dp;
}

/*---------------------------------------------------------------------*/


void print_matrix(int **matrix, int n) {
    int max_val = 0;

    // Cherche la valeur maximale pour adapter l'espacement
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            int val = abs(matrix[i][j]);
            if (val > max_val) {
                max_val = val;
            }
        }
    }

    // Calcule le nombre de chiffres de la plus grande valeur
    int width = 1;
    while (max_val /= 10)
        width++;
    width += 2; // Ajoute un peu d'espace pour l'alignement et les signes

    // Affiche la matrice avec le bon format
    for (int i = 0; i < n; i++) {
        printf("[ ");
        for (int j = 0; j < n; j++) {
            printf("%*d ", width, matrix[i][j]);
        }
        printf("]\n");
    }
}

/*---------------------------------------------------------------------*/


tab init_tab_null()
{
    tab t;
    t.d0 = -1;
    t.d1 = -1;
    t.d2 = -1;
    t.d3 = -1;
    t.tab=NULL;
    return t;
}

/*---------------------------------------------------------------------*/


double*** charger_tableaux() 
{
    FILE *fp = fopen("formes.txt", "r");
    int n;
    fscanf(fp, "%d\n", &n);
    double ***formes = malloc(n * sizeof(double**));
    char line[2048];
    for (int i = 0; i < n; i++) 
    {
        fgets(line, sizeof(line), fp);
        while (line[0] == '\n') fgets(line, sizeof(line), fp);  // saute les lignes vides
        if (strncmp(line, "NULL", 4) == 0) 
        {
            formes[i] = NULL;
        } 
        else 
        {
            formes[i] = malloc(28 * sizeof(double*));
            formes[i][0] = malloc(28 * 28 * sizeof(double));

            for (int j = 0; j < 28; j++) 
            {
                if (j > 0) fgets(line, sizeof(line), fp);
                formes[i][j] = formes[i][0] + j * 28;
                char *token = strtok(line, " ");
                for (int k = 0; k < 28; k++) 
                {
                    formes[i][j][k] = atof(token);
                    token = strtok(NULL, " ");
                }
            }
        }
    }

    fclose(fp);
    return formes;
}

/*---------------------------------------------------------------------*/


void save_model(cnn* net, char* filename) 
{
    FILE* f = fopen(filename, "w");
    // Écrire nb_step et spmb
    fprintf(f, "%d\n", net->nb_step);
    fprintf(f, "%d\n", net->spmb);
    for (int st = 0; st < net->nb_step; st++) 
    {
        layer* l = &net->step[st];

        // Type de la couche
        fprintf(f, "%d\n", l->type);

        // Dimensions de z
        fprintf(f, "%d %d %d %d\n", l->z.d0, l->z.d1, l->z.d2, l->z.d3);

        // Si la couche a des poids
        if (l->w.tab != NULL) 
        {
            fprintf(f, "1\n"); // w présent
            fprintf(f, "%d %d %d %d\n", l->w.d0, l->w.d1, l->w.d2, l->w.d3);
            for (int i = 0; i < l->w.d0; i++)
            for (int j = 0; j < l->w.d1; j++)
            for (int k = 0; k < l->w.d2; k++)
            for (int m = 0; m < l->w.d3; m++)
            fprintf(f, "%.15lf ", l->w.tab[i][j][k][m]);
            fprintf(f, "\n");
        } 
        else 
        {
            fprintf(f, "0\n"); // w absent
        }

        // Si la couche a des biais
        if (l->b.tab != NULL) 
        {
            fprintf(f, "1\n"); // b présent
            fprintf(f, "%d %d %d %d\n", l->b.d0, l->b.d1, l->b.d2, l->b.d3);
            for (int i = 0; i < l->b.d0; i++)
            for (int j = 0; j < l->b.d1; j++)
            for (int k = 0; k < l->b.d2; k++)
            for (int m = 0; m < l->b.d3; m++)
            fprintf(f, "%.15lf ", l->b.tab[i][j][k][m]);
            fprintf(f, "\n");
        } 
        else 
        {
            fprintf(f, "0\n"); // b absent
        }
    }

    fclose(f);
}

/*---------------------------------------------------------------------*/


cnn* load_model(char* filename) 
{
    FILE* f = fopen(filename, "r");
    int nb_step, spmb;
    fscanf(f, "%d", &nb_step);
    fscanf(f, "%d", &spmb);

    cnn* net = malloc(sizeof(cnn));
    net->nb_step = nb_step;
    net->spmb = spmb;
    net->step = malloc(nb_step * sizeof(layer));
    net->adam_t = 0;

    for (int st = 0; st < nb_step; st++) 
    {
        layer* l = &net->step[st];
        int type;
        fscanf(f, "%d", &type);
        l->type = (type_de_couche)type;

        // Lecture des dimensions de z
        fscanf(f, "%d %d %d %d", &l->z.d0, &l->z.d1, &l->z.d2, &l->z.d3);
        l->z.tab = malloc(sizeof(double***) * l->z.d0);
        for (int i = 0; i < l->z.d0; i++) 
        {
            l->z.tab[i] = malloc(sizeof(double**) * l->z.d1);
            for (int j = 0; j < l->z.d1; j++) 
            {
                l->z.tab[i][j] = malloc(sizeof(double*) * l->z.d2);
                for (int k = 0; k < l->z.d2; k++) 
                {
                    l->z.tab[i][j][k] = calloc(l->z.d3, sizeof(double));
                }
            }
        }

        // Lecture de w si présent
        int has_w;
        fscanf(f, "%d", &has_w);
        if (has_w) 
        {
            fscanf(f, "%d %d %d %d", &l->w.d0, &l->w.d1, &l->w.d2, &l->w.d3);
            l->w.tab = malloc(sizeof(double***) * l->w.d0);
            for (int i = 0; i < l->w.d0; i++) 
            {
                l->w.tab[i] = malloc(sizeof(double**) * l->w.d1);
                for (int j = 0; j < l->w.d1; j++) 
                {
                    l->w.tab[i][j] = malloc(sizeof(double*) * l->w.d2);
                    for (int k = 0; k < l->w.d2; k++) 
                    {
                        l->w.tab[i][j][k] = malloc(sizeof(double) * l->w.d3);
                        for (int m = 0; m < l->w.d3; m++) 
                        {
                            fscanf(f, "%lf", &l->w.tab[i][j][k][m]);
                        }
                    }
                }
            }
        } 
        else 
        {
            l->w.tab = NULL;
        }

        // Lecture de b si présent
        int has_b;
        fscanf(f, "%d", &has_b);
        if (has_b) 
        {
            fscanf(f, "%d %d %d %d", &l->b.d0, &l->b.d1, &l->b.d2, &l->b.d3);
            l->b.tab = malloc(sizeof(double***) * l->b.d0);
            for (int i = 0; i < l->b.d0; i++) 
            {
                l->b.tab[i] = malloc(sizeof(double**) * l->b.d1);
                for (int j = 0; j < l->b.d1; j++) 
                {
                    l->b.tab[i][j] = malloc(sizeof(double*) * l->b.d2);
                    for (int k = 0; k < l->b.d2; k++) 
                    {
                        l->b.tab[i][j][k] = malloc(sizeof(double) * l->b.d3);
                        for (int m = 0; m < l->b.d3; m++) 
                        {
                            fscanf(f, "%lf", &l->b.tab[i][j][k][m]);
                        }
                    }
                }
            }
        } 
        else 
        {
            l->b.tab = NULL;
        }

        // Les autres champs restent vides, ils ne servent qu'a l'entrainement et je ne compte pas save un modele puis le re entrainer
        l->dropout.tab = NULL;
        l->sdw.tab = NULL;
        l->sdb.tab = NULL;
        l->vdw.tab = NULL;
        l->vdb.tab = NULL;
        l->adam_t = 0;
    }

    fclose(f);
    return net;
}

/*---------------------------------------------------------------------*/


int main_entrainement() 
{
  load_mnist();
  for(int i = 0 ; i < NUM_TRAIN ; i++)
  {
    train_label[i]--;
  }
  for(int i = 0 ; i < NUM_TEST ; i++)
  {
    test_label[i]--;
  }
  double acc=0;
  int N = 10;
  int c=0;
  type_de_couche types[] = {input,conv,max_pool,flat,fc,relu,fc,relu,fc,output};
  assert(types[N-1]==output);
  int dim_poids[] = {0,3,0,0,512,0,128,0,OUTPUT_SIZE,0}; //0 pour les couches sans poids
  int dim_entree = 28;
  int dim[] = {1,32,32,32}; //pas besoin de plus car une fois dans la fully_connected on ne prend pas en compte les dimensions
  int spmb = 8; //taille des minis batchs (j'ai 8 threads sur mon ordi donc j'en met 8)
  int nb_max = -1;
  cnn* net = init(N, types, dim, dim_entree, dim_poids,spmb);
  net->adam_t=0;
  int* exp = malloc(spmb*sizeof(int));
  tab* error = malloc(net->nb_step*sizeof(tab));
  int compteur_precision = 0;
  for(int i = 0 ; i < net->nb_step ; i++)
  {
    error[i] = init_tab_dim_4(net->spmb,net->step[i].z.d1,net->step[i].z.d2,net->step[i].z.d3);
  }
  int nb_epoch = 10; 
  for(int k = 0 ; k < nb_epoch ; k++)
  {
    for(int a = 0 ; a < NUM_TRAIN - NUM_TRAIN%spmb; a+=spmb)
    {
        for(int mb = 0 ; mb < net->spmb ; mb++)
        {
            for(int i = 0 ; i < dim_entree ; i++)
            {
                for(int j = 0 ; j < dim_entree ; j++)
                {
                    net->step[0].z.tab[mb][0][i][j] = train_image[a+mb][i*dim_entree+j];
                }
            }
            if(train_label[a+mb] < nb_sep1 )exp[mb] = 0;
            else if (train_label[a+mb] >= nb_sep2)exp[mb] = 1;
            else exp[mb]= 2;
        }
        feedforward_en_parallele(net);
        backprop_en_parallele(net,error,exp,&acc,&compteur_precision);
        update(net,error);
        c+=spmb;
        if(c==1000)
        {
            c=0;
            printf("%f %%  de l'epoch numero %d sur %d : \n coût moyen : %f | précision sur les 1000 derniers : %f\n",
                ((double)a)/(NUM_TRAIN - NUM_TRAIN%spmb),k,nb_epoch,((double)(acc/(1000))),((double)compteur_precision/((double)1000)));
            fflush(stdout);//sinon ca ne se print qu'à la fin de la boucle...
            if(compteur_precision>nb_max)//si on a un meilleur model
            {
                save_model(net,"save.txt");
                nb_max = compteur_precision;
            }
            compteur_precision=0;
            acc=0;
        }
    }
    }
   //test du réseau le plus performant
  int** matrice_confu = malloc(OUTPUT_SIZE*sizeof(int*));
  for(int i = 0 ; i < OUTPUT_SIZE ; i++)
  {
    matrice_confu[i] = malloc(OUTPUT_SIZE * sizeof(int*));
    for(int j = 0 ; j < OUTPUT_SIZE ; j++) matrice_confu[i][j] = 0;
  }
  int compteur = 0;
  net = load_model("save.txt");
  for(int a = 0 ; a < NUM_TEST - NUM_TEST%spmb; a+=spmb)
  {
    for(int mb = 0 ; mb < net->spmb ; mb++)
    {
        for(int i = 0 ; i < dim_entree ; i++)
        {
            for(int j = 0 ; j < dim_entree ; j++)
            {
                net->step[0].z.tab[mb][0][i][j] = test_image[a+mb][i*dim_entree+j];
            }
        }
    }
    feedforward_en_parallele(net);
    for(int mb = 0 ; mb < net->spmb ; mb++)
    {
        double max = net->step[net->nb_step - 1].z.tab[mb][0][0][0];
        int i_max = 0;
        for(int i = 1 ; i < OUTPUT_SIZE ; i++)
        {
            if(net->step[net->nb_step - 1].z.tab[mb][0][i][0]>max)
            {
                max = net->step[net->nb_step - 1].z.tab[mb][0][i][0];
                i_max = i;
            }
        }
        int good_shot = -1;
        if(test_label[a+mb] < nb_sep1 )good_shot = 0;
        else if (test_label[a+mb] >= nb_sep2)good_shot = 1;
        else good_shot= 2;
        matrice_confu[test_label[a+mb]][i_max]++;
        if(i_max == good_shot) compteur++;
    }
  }
  print_matrix(matrice_confu,OUTPUT_SIZE);
  printf("résultats finaux :  %d sur %d",compteur,NUM_TEST - NUM_TEST%spmb);
  return 0;
}

/*---------------------------------------------------------------------*/


char asci(int asci_code)
{
    switch (asci_code) 
    {
        case 48: return '0';
        case 49: return '1';
        case 50: return '2';
        case 51: return '3';
        case 52: return '4';
        case 53: return '5';
        case 54: return '6';
        case 55: return '7';
        case 56: return '8';
        case 57: return '9';
        case 65: return 'A';
        case 66: return 'B';
        case 67: return 'C';
        case 68: return 'D';
        case 69: return 'E';
        case 70: return 'F';
        case 71: return 'G';
        case 72: return 'H';
        case 73: return 'I';
        case 74: return 'J';
        case 75: return 'K';
        case 76: return 'L';
        case 77: return 'M';
        case 78: return 'N';
        case 79: return 'O';
        case 80: return 'P';
        case 81: return 'Q';
        case 82: return 'R';
        case 83: return 'S';
        case 84: return 'T';
        case 85: return 'U';
        case 86: return 'V';
        case 87: return 'W';
        case 88: return 'X';
        case 89: return 'Y';
        case 90: return 'Z';
        case 97: return 'a';
        case 98: return 'b';
        case 100: return 'd';
        case 101: return 'e';
        case 102: return 'f';
        case 103: return 'g';
        case 104: return 'h';
        case 110: return 'n';
        case 113: return 'q';
        case 114: return 'r';
        case 116: return 't';
        default: return '?'; // Si le code asci n'est pas dans la table
    }
}

/*---------------------------------------------------------------------*/


void feedforward_sans_thread(cnn* net)
{
    for(int a = 0 ; a < net->spmb ; a++)
    {
        for(int st = 0 ; st < net->nb_step ; st++)
            {
                layer* l = &net->step[st];
                switch(l->type)
                {
                    case input:
                    {
                        break;
                    }
                    case output:
                    {
                        for(int i = 0 ; i < net->step[st-1].z.d1 ; i++)
                        {
                            double max = net->step[st-1].z.tab[a][i][0][0];
                            for(int h = 1 ; h < net->step[st-1].z.d2 ; h++)
                            {
                                if(net->step[st-1].z.tab[a][i][h][0]>max)max=net->step[st-1].z.tab[a][i][h][0];
                            }
                            //on calcule la somme des exp pour le softmax :
                            double c = 0;
                            for(int j = 0 ; j < net->step[st-1].z.d2 ; j++) 
                            {
                                c+=exp(net->step[st-1].z.tab[a][i][j][0] - max);
                            }
                            assert(!isnan(c));
                            for(int j = 0 ; j < net->step[st-1].z.d2 ; j++)
                            {
                                l->z.tab[a][i][j][0] = exp((net->step[st-1].z.tab[a][i][j][0]) - max)/c;
                            }
                        }
                        break;
                    }
                    case max_pool:
                    {
                        for(int i = 0 ; i < l->z.d1 ; i++)
                        {
                            for(int j = 0 ; j < l->z.d2 ; j++)
                            {
                                for(int k = 0 ; k < l->z.d3 ; k++)
                                {
                                    l->z.tab[a][i][j][k]=max_tab_2(net->step[st-1].z.tab[a][i],j*2,k*2);
                                }
                            }
                        }
                        break;
                    }
                    case avg_pool:
                    {
                        for(int i = 0 ; i < l->z.d1 ; i++)
                        {
                            for(int j = 0 ; j < l->z.d2 ; j++)
                            {
                                for(int k = 0 ; k < l->z.d3 ; k++)
                                {
                                    l->z.tab[a][i][j][k]=avg_tab_2(net->step[st-1].z.tab[a][i],j*2,k*2);
                                }
                            }
                        }
                        break;
                    }
                    case conv:
                    {
                        for(int i = 0 ; i < l->w.d0 ; i++)
                        {
                            for(int j = 0 ; j < l->z.d2 ; j++)
                            {
                                for(int k = 0 ; k < l->z.d3; k++)
                                {
                                    double c = 0;
                                        for(int i1 = 0 ; i1 < l->w.d1 ; i1++)//on somme sur toute les dimensions
                                        {
                                            for(int j1 = 0 ; j1 < l->w.d2 ; j1++)
                                            {
                                                for(int k1 = 0 ; k1 < l->w.d2 ; k1++)
                                                {
                                                    c+=l->w.tab[i][i1][j1][k1]*net->step[st-1].z.tab[a][i1][j+j1][k+k1];
                                                }
                                            }
                                        }
                                        c+=l->b.tab[0][0][0][i];
                                        l->z.tab[a][i][j][k] = c;
                                }
                            }
                            
                        }
                        break;
                    }
                    case flat:
                    {
                        for(int i = 0 ; i < net->step[st-1].z.d1 ; i++)
                        {
                            for(int j = 0 ; j < net->step[st-1].z.d2 ; j++)
                            {
                                for(int k = 0 ; k < net->step[st-1].z.d3 ; k++)
                                {
                                    l->z.tab[a][0][i*net->step[st-1].z.d2*net->step[st-1].z.d3+
                                                    j*net->step[st-1].z.d3+k][0] 
                                    = 
                                    net->step[st-1].z.tab[a][i][j][k];
                                }
                            }
                        }
                        break;
                    }
                    case relu:
                    {
                        for(int i = 0 ; i < net->step[st-1].z.d1 ; i++)
                        {
                            for(int j = 0 ; j < net->step[st-1].z.d2 ; j++)
                            {
                                for(int k = 0 ; k < net->step[st-1].z.d3 ; k++)
                                {
                                    l->z.tab[a][i][j][k] = apply_relu(net->step[st-1].z.tab[a][i][j][k]);
                                }
                            }
                        }
                        break;
                    }
              
                    case fc:
                    {
                        for(int i = 0 ; i < l->z.d2 ; i++)
                        {
                            double c = 0;
                                for(int j = 0 ; j < net->step[st-1].z.d2 ; j++)
                                {
                                    c+=l->w.tab[0][0][i][j]*net->step[st-1].z.tab[a][0][j][0];
                                }
                                c+=l->b.tab[0][0][0][i];
                                l->z.tab[a][0][i][0] = c;
                        }
                        break;
                    }
                }
            }
    }
}

/*---------------------------------------------------------------------*/


void backprop_sans_threads(cnn* net,tab* error, int* expected, double* acc,int* compteur_precision)
{
    //calcul de l'erreur
    for(int a = 0 ; a < net->spmb ; a++)
    {
        int i_max = 0;
        double max = -INFINITY;
        for(int i = 0 ; i < net->step[net->nb_step - 1].z.d2 ; i++)
        {
            if(net->step[net->nb_step - 1].z.tab[a][0][i][0] > max)
            {
                max=net->step[net->nb_step - 1].z.tab[a][0][i][0];
                i_max=i;
            }
            if(i == expected[a])
            {
                error[net->nb_step - 1].tab[a][0][i][0] = net->step[net->nb_step - 1].z.tab[a][0][i][0]-1;
                (*acc)-=log(net->step[net->nb_step - 1].z.tab[a][0][i][0]+eps);
            } 
            else
            {
                error[net->nb_step - 1].tab[a][0][i][0] = net->step[net->nb_step - 1].z.tab[a][0][i][0];  
            }
        }  
        if(i_max==expected[a])(*compteur_precision)++;
    }
    for(int st = net->nb_step - 2 ; st > -1 ; st--)
    {
        tab* p = &error[st+1];
        tab* l = &error[st];
        switch(net->step[st+1].type)
        {
            case input:
            {
                assert(1==1);
                break;
            }
            case output:
            {
                for(int a = 0 ; a < net->spmb; a++)
                {
                    for(int i = 0 ; i < l->d1 ; i++)
                    {
                        for(int j = 0 ; j < l->d2 ; j++)
                        {
                            for(int k = 0 ; k < l->d3 ; k++)
                            {
                               l->tab[a][i][j][k] = p->tab[a][i][j][k];
                               
                            }
                        }
                    }
                }
                break;
            }
            case max_pool:
            {
                for(int a = 0 ; a < net->spmb; a++)
                {
                    for(int i = 0 ; i < l->d1 ; i++)
                    {
                        for(int j = 0 ; j < l->d2 ; j++)
                        {
                            for(int k = 0 ; k < l->d3 ; k++)
                            {
                               if(k/2 < net->step[st+1].z.d3 && j/2 < net->step[st+1].z.d2)
                               {
                                    if(net->step[st].z.tab[a][i][j][k] == net->step[st+1].z.tab[a][i][j/2][k/2])//si c'est le max qui a été choisi par le pool
                                    {
                                        l->tab[a][i][j][k] = p->tab[a][i][j/2][k/2];
                                    }
                                    else 
                                    {
                                        l->tab[a][i][j][k] = 0;
                                    }
                               }
                            }
                        }
                    }
                }
                break;
            }
            case avg_pool:
            {
                for(int a = 0 ; a < l->d0; a++)
                {
                    for(int i = 0 ; i < l->d1 ; i++)
                    {
                        for(int j = 0 ; j < l->d2 ; j++)
                        {
                            for(int k = 0 ; k < l->d3 ; k++)
                            {
                                l->tab[a][i][j][k] = p->tab[a][i][j/2][k/2];
                            }
                        }
                    }
                }
                break;
            }
            case conv:
            {   
                for(int a = 0 ; a < net->spmb ; a++)
                {
                    for(int i = 0 ; i < net->step[st].z.d1 ; i++)
                    {
                        for(int j = 0 ; j < net->step[st].z.d2 ; j++)
                        {
                            for(int k = 0 ; k < net->step[st].z.d3 ; k++)
                            {
                                double c = 0;
                                for(int i1 = 0 ; i1 < net->step[st+1].z.d1 ; i1++)
                                {
                                    for(int j1 = 0 ; j1 < net->step[st+1].w.d2 ; j1++)
                                    {
                                        for(int k1 = 0 ; k1 < net->step[st+1].w.d2 ; k1++)
                                        {
                                            if(j-j1 > -1 && j-j1 < net->step[st+1].z.d2 &&
                                                k-k1 > -1 && k-k1 < net->step[st+1].z.d3)
                                                {
                                                    c+=p->tab[a][i1][j-j1][k-k1]*
                                                        net->step[st+1].w.tab[i1][i][j1][k1];
                                                }
                                        }
                                    }
                                }
                                l->tab[a][i][j][k] = c;
                            }
                        }
                    }
                }
                break;
            }
            case flat:
            {
                for(int a = 0 ; a < net->spmb; a++)
                {
                    for(int i = 0 ; i < l->d1 ; i++)
                    {
                        for(int j = 0 ; j < l->d2 ; j++)
                        {
                            for(int k = 0 ; k < l->d3 ; k++)
                            {
                                l->tab[a][i][j][k] = p->tab[a][0][i*l->d2*l->d3+j*l->d3+k][0];
                            }
                        }
                    }
                }
                break;
            }
            case relu:
            {
                for(int a = 0 ; a < net->spmb; a++)
                {
                    for(int i = 0 ; i < l->d1 ; i++)
                    {
                        for(int j = 0 ; j < l->d2 ; j++)
                        {
                            for(int k = 0 ; k < l->d3 ; k++)
                            {
                               l->tab[a][i][j][k] =p->tab[a][i][j][k]* apply_relu_der(net->step[st].z.tab[a][i][j][k]);
                            }
                        }
                    }
                }
                break;
            }
            case fc:
            {
                for(int a = 0 ; a < net->spmb ; a++)
                {
                    
                    for(int i = 0 ; i < net->step[st].z.d2 ; i++)
                    {
                        double c = 0;
                        for(int j = 0 ; j < net->step[st+1].z.d2 ; j++)
                        {
                            c+=net->step[st+1].w.tab[0][0][j][i]*
                                p->tab[a][0][j][0];
                        }
                        l->tab[a][0][i][0] = c;
                        
                    }
                }
                break;
            }
        }
    }
    //modification des paramètres
    double last;
    net->adam_t++;
    double beta_t1 = pow(beta1,net->adam_t);
    double beta_t2=pow(beta2,net->adam_t);
    for(int st = 0 ; st < net->nb_step ; st++)
    {
        layer* lay = &net->step[st];
        if(net->step[st].type == conv)
        {
            //poids
            for(int i = 0 ; i < net->step[st].w.d0 ; i++)
            {
                for(int j = 0 ; j < net->step[st].w.d1 ; j++)
                {
                    for(int k = 0 ; k < net->step[st].w.d2 ; k++)
                    {
                        for(int l = 0 ; l < net->step[st].w.d2 ; l++)
                        {
                            double c = 0;
                            for(int a = 0 ; a < net->spmb ; a++)
                            {
                                for(int k1 = 0 ; k1 < error[st].d2 ; k1++)
                                {
                                    for(int l1 = 0 ; l1 < error[st].d3 ; l1++)
                                    {
                                        if(k+k1 < net->step[st-1].z.d2 &&
                                            l+l1 < net->step[st-1].z.d3)
                                        {
                                            c+=net->step[st-1].z.tab[a][j][k+k1][l+l1]*
                                                error[st].tab[a][i][k1][l1];
                                        }
                                    }
                                }
                            }
                            double cp = c/net->spmb;
                            last = lay->sdw.tab[i][j][k][l];
                            lay->sdw.tab[i][j][k][l] = beta1*last+(1-beta1)*cp;
                            double m = lay->sdw.tab[i][j][k][l]/((1-beta_t1) );

                            last = lay->vdw.tab[i][j][k][l];
                            lay->vdw.tab[i][j][k][l] = beta2*last+(1-beta2)*cp*cp;
                            double v =  lay->vdw.tab[i][j][k][l]/(1-beta_t2);
                            lay->w.tab[i][j][k][l]-=LR*m/(sqrt(v+eps));
                        }
                    }
                }
            }
            //biais
            for(int i = 0 ; i < net->step[st].b.d3 ; i++)
            {
                double c = 0;
                for(int a = 0 ; a < net->spmb ; a++)
                {
                    for(int j = 0 ; j < error[st].d2 ; j++)
                    {
                        for(int k = 0 ; k < error[st].d3 ; k++)
                        {
                            c+=error[st].tab[a][i][j][k];
                        }
                    }
                }
                double cp = c/net->spmb;
                last = lay->sdb.tab[0][0][0][i];
                lay->sdb.tab[0][0][0][i] = beta1*last+(1-beta1)*cp;
                double m = lay->sdb.tab[0][0][0][i]/(1-beta_t1);

                last = lay->vdb.tab[0][0][0][i];
                lay->vdb.tab[0][0][0][i] = beta2*last+(1-beta2)*cp*cp;
                double v = lay->vdb.tab[0][0][0][i]/(1-beta_t2);
                lay->b.tab[0][0][0][i] -= LR*m/sqrt(v+eps);
            }
        }
        else if (net->step[st].type == fc)
        {
            //poids
            for(int i = 0 ; i < net->step[st].w.d2 ; i++)
            {
                for(int j = 0 ; j < net->step[st].w.d3 ; j++)
                {
                    double c = 0;
                    for(int a = 0 ; a < net->spmb ; a++)
                    {
                        c+=error[st].tab[a][0][i][0]*
                            net->step[st-1].z.tab[a][0][j][0];
                    }
                    double cp = c/net->spmb; 
                    last = lay->sdw.tab[0][0][i][j];
                    lay->sdw.tab[0][0][i][j] = beta1*last+(1-beta1)*cp;
                    double m = lay->sdw.tab[0][0][i][j]/((1-beta_t1));

                    last = lay->vdw.tab[0][0][i][j];
                    lay->vdw.tab[0][0][i][j] = beta2*last+(1-beta2)*cp*cp;
                    double v =  lay->vdw.tab[0][0][i][j]/(1-beta_t2);
                    lay->w.tab[0][0][i][j]-=LR*m/(sqrt(v+eps));
                }
            }
            //biais
            for(int i = 0 ; i < net->step[st].b.d3 ; i++)
            {
                double c = 0;
                for(int a = 0 ; a < net->spmb ; a++)
                {
                    c+=error[st].tab[a][0][i][0];
                }
                double cp = c/net->spmb;
                last = lay->sdb.tab[0][0][0][i];
                lay->sdb.tab[0][0][0][i] = beta1*last+(1-beta1)*cp;
                double m = lay->sdb.tab[0][0][0][i]/(1-beta_t1);

                last = lay->vdb.tab[0][0][0][i];
                lay->vdb.tab[0][0][0][i] = beta2*last+(1-beta2)*cp*cp;
                double v = lay->vdb.tab[0][0][0][i]/(1-beta_t2);
                lay->b.tab[0][0][0][i] -= LR*m/sqrt(v+eps); 
            }
        }
    }
    //reinitialisation du tableau erreur
    for(int st = 0 ; st < net->nb_step ; st++)
    {
        for(int i = 0 ; i < error[st].d0 ; i++)
        {
            for(int j = 0 ; j < error[st].d1 ; j++)
            {
                for(int k = 0 ; k < error[st].d2 ; k++)
                {
                    for(int l = 0 ; l < error[st].d3 ; l++)
                    {
                        error[st].tab[i][j][k][l] = 0;
                    }
                }
            }
        }
    }
}

/*---------------------------------------------------------------------*/


int main_reconnaissance()
{
    load_mnist();
    cnn* net_numbers = load_model("save_reseau_numbers.txt");
    cnn* net_majuscules = load_model("save_reseau_select_majuscules.txt");
    cnn* net_minuscules = load_model("cnn_with_multi_select_minuscules_95.txt");
    cnn* reconaissance = load_model("save_reseau_classifieur_en_3_86.txt");
    int nb_images;
    FILE* f = fopen("formes.txt", "r");
    fscanf(f, "%d\n", &nb_images);
    double*** tab = charger_tableaux();
    for(int a = 0 ; a < nb_images; a++)
    {
        if(tab[a]==NULL) printf("espace\n");
        else
        {
            for(int i = 0 ; i < 28 ; i++)
            {
                for(int j = 0 ; j < 28 ; j++)
                {
                    reconaissance->step[0].z.tab[0][0][i][j] = tab[a][i][j];
                }
            }
            reconaissance->spmb=1;
            feedforward_sans_thread(reconaissance);
            double max = reconaissance->step[reconaissance->nb_step - 1].z.tab[0][0][0][0];
            int i_max = 0;
            for(int i = 1 ; i < 3 ; i++)
            {
                if(reconaissance->step[reconaissance->nb_step - 1].z.tab[0][0][i][0]>max)
                {
                    max = reconaissance->step[reconaissance->nb_step - 1].z.tab[0][0][i][0];
                    i_max = i;
                }
            }
            printf("classifié comme un %d\n",i_max);
            if(i_max == 0)
            {
                
                int new_imax = 0;
                for(int i = 0 ; i < 28 ; i++)
                {
                    for(int j = 0 ; j < 28 ; j++)
                    {
                        net_numbers->step[0].z.tab[0][0][i][j] = tab[a][i][j];
                    }
                }
                net_numbers->spmb=1;
                feedforward_sans_thread(net_numbers);
                double max = net_numbers->step[net_numbers->nb_step - 1].z.tab[0][0][0][0];
                for(int i = 1 ; i < 26 ; i++)
                {
                    if(net_numbers->step[net_numbers->nb_step - 1].z.tab[0][0][i][0]>max)
                    {
                        max = net_numbers->step[net_numbers->nb_step - 1].z.tab[0][0][i][0];
                        new_imax = i;
                    }
                }
                printf("c'est un %c\n",asci(new_imax));
            }
            if(i_max == 1)
            {
                int new_imax = 0;
                for(int i = 0 ; i < 28 ; i++)
                {
                    for(int j = 0 ; j < 28 ; j++)
                    {
                        net_minuscules->step[0].z.tab[0][0][i][j] = tab[a][i][j];
                    }
                }
                net_minuscules->spmb=1;
                feedforward_sans_thread(net_minuscules);
                double max = net_minuscules->step[net_minuscules->nb_step - 1].z.tab[0][0][0][0];
                for(int i = 1 ; i < 11 ; i++)
                {
                    if(net_minuscules->step[net_minuscules->nb_step - 1].z.tab[0][0][i][0]>max)
                    {
                        max = net_minuscules->step[net_minuscules->nb_step - 1].z.tab[0][0][i][0];
                        new_imax = i;
                    }
                }
                printf("c'est un %c\n",asci(new_imax+36));
                
            }
            if(i_max == 2)
            {
                
                int new_imax = 0;
                for(int i = 0 ; i < 28 ; i++)
                {
                    for(int j = 0 ; j < 28 ; j++)
                    {
                        net_majuscules->step[0].z.tab[0][0][i][j] = tab[a][i][j];
                    }
                }
                net_majuscules->spmb=1;
                feedforward_sans_thread(net_majuscules);
                double max = net_majuscules->step[net_majuscules->nb_step - 1].z.tab[0][0][0][0];
                for(int i = 1 ; i < 26 ; i++)
                {
                    if(net_majuscules->step[net_majuscules->nb_step - 1].z.tab[0][0][i][0]>max)
                    {
                        max = net_majuscules->step[net_majuscules->nb_step - 1].z.tab[0][0][i][0];
                        new_imax = i;
                    }
                }
                printf("c'est un %c\n",asci(new_imax+10));
            }
        }
    }
    printf("fin du mot");
    return 0;
}
int main()
{
    printf("utiliser main_entrainement pour l'entrainement et main_reconnaissance pour le test");
    return 0;
}
