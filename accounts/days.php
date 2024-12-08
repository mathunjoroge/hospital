<?php 
$adm_date=date("Y-m-d", strtotime($row['adm_date']));
$date1=date_create($adm_date);
$date2=date_create($discharge_date);
$diff=date_diff($date1,$date2);
// calculate number of days
function dateDiff ($discharge_date, $adm_date) {

    // Return the number of days between the two dates:    
    return round(abs(strtotime($discharge_date) - strtotime($adm_date))/86400);

}
$days= round(abs(strtotime($discharge_date) - strtotime($adm_date))/86400);
if ($days === null) {
    // Set default value for days if it is null
    $days = 0;
}
// end calcalate days
?> 