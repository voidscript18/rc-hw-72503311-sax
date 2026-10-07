# Linux 常用命令笔记

- pwd：看我现在在哪个目录
- ls：列出当前目录的文件，ls -lh 能看大小，ls -a 连隐藏文件一起看
- cd：切换目录，cd .. 回上一级，cd ~ 回家目录，cd - 回上一个待过的地方
- mkdir：新建文件夹，mkdir -p a/b/c 一次建好几层
- cp：复制，cp a.txt b/ 复制文件，cp -r 复制整个文件夹
- mv：移动或改名，mv old.py new.py
- rm：删除，rm -rf 删文件夹，删了就找不回来，要小心
- cat：直接打印文件内容
- head：看文件开头几行，head -5 a.txt 看前 5 行
- tail：看文件末尾几行，tail -5 a.txt 看最后 5 行
- grep：在文件里搜关键字，grep -n error log.txt 会带行号
- find：找文件，find . -name "*.py"
- sudo：以管理员身份执行（Ubuntu 装软件用，Windows 上我用 pip install 代替）
- chmod：改权限，chmod +x run.sh 让脚本能直接执行
- df：看磁盘还剩多少空间，df -h
- ping：测网络通不通，ping baidu.com
- tar：打包和解压，tar -xzf a.tar.gz 解压

## 另外三条保命的

- history：看我之前敲过什么命令
- Tab 键：自动补全，省一半打字
- Ctrl+C：程序卡死时强制停掉
