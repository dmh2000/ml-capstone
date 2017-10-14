wget https://dds.cr.usgs.gov/srtm/version2_1/SRTM3/North_America/N37W098.hgt.zip
unzip N37W098.hgt.zip
mv -f N37W098.hgt level1/N37W098.hgt
rm N37W098.hgt.zip

wget https://dds.cr.usgs.gov/srtm/version2_1/SRTM3/North_America/N39W120.hgt.zip
unzip N39W120.hgt.zip
mv -f N39W120.hgt level1/N39W120.hgt
rm N39W120.hgt.zip

wget https://dds.cr.usgs.gov/srtm/version2_1/SRTM1/Region_05/N37W098.hgt.zip
unzip N37W098.hgt.zip
mv -f N37W098.hgt level2/N37W098.hgt
rm N37W098.hgt.zip

wget https://dds.cr.usgs.gov/srtm/version2_1/SRTM1/Region_01/N39W120.hgt.zip
unzip N39W120.hgt.zip
mv -f N39W120.hgt level2/N39W120.hgt
rm N39W120.hgt.zip

