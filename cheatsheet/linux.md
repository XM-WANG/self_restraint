## Linux

1. Check CPU version
```
cat /proc/cpuinfo | grep 'model name' |uniq
```

2. Check disk quota
```
du -sh ./
```

3. ln

```
ln -s src des
ln -s -F src_folder/ des_folder
unlink des
```
