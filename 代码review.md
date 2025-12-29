1. aux-loss，在mainep和moelayer.moe.fwd里面是明显重复的，怎么删掉？

新增一块 moe-ep-layer用于实现class EPMoE，而不是在mainep这个训练脚本里实现aux-loss和含ata调用的MoE。现在把功能解耦开，让结构更加清晰。

然后为了想评估下，训练速度，新增了时间日志，在log-interval打印loss的时候也打印训练需要的时间。

