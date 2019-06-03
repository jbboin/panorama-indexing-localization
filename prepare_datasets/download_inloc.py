import os, subprocess, tarfile, zipfile
import config

dest_dir = os.path.join(config.INLOC_ROOT, 'download')

if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

# Download queries from InLoc dataset: http://www.ok.sc.e.titech.ac.jp/INLOC/subpages/inloc_query.html
query_url = 'http://www.ok.sc.e.titech.ac.jp/INLOC/materials/iphone7.tar.gz'
cmd = 'wget %s -P %s' % (query_url, dest_dir)
subprocess.call(cmd, shell=True)
query_tar = os.path.join(dest_dir, query_url.split('/')[-1])
tar = tarfile.open(query_tar)
tar.extractall(path=dest_dir)
tar.close()

# Download panoramas from InLoc dataset (format is easier to process than the original data from the Wustl Dataset)
# http://www.ok.sc.e.titech.ac.jp/INLOC/subpages/inloc_db.html
db_url = 'http://www.ok.sc.e.titech.ac.jp/INLOC/materials/scans.tar.gz'
cmd = 'wget %s -P %s' % (db_url, dest_dir)
subprocess.call(cmd, shell=True)
db_tar = os.path.join(dest_dir, db_url.split('/')[-1])
tar = tarfile.open(db_tar)
tar.extractall(path=dest_dir)
tar.close()

for building in ['DUC1', 'DUC2']:
    tar = tarfile.open(os.path.join(dest_dir, 'scans', building+'.tar'))
    tar.extractall(path=os.path.join(dest_dir, 'scans'))
    tar.close()

# Download transformations from Wustl Indoor RGBD Dataset: https://cvpr17.wijmans.xyz/data/
align_dir = os.path.join(dest_dir, 'alignment')
build_urls = {
    'DUC1':'https://www.dropbox.com/s/v4fh8qo5ec03av1/DUC1_alignment.zip',
    'DUC2':'https://www.dropbox.com/s/vzzvs8z7ytjt33k/DUC2_alignment.zip'
}
for b in build_urls:
    cmd = 'wget %s -P %s' % (build_urls[b], align_dir)
    subprocess.call(cmd, shell=True)
    duc_zip = os.path.join(align_dir, build_urls[b].split('/')[-1])
    zip_ref = zipfile.ZipFile(duc_zip, 'r')
    zip_ref.extractall(path=os.path.join(align_dir, b))
    zip_ref.close()

