
<?php
$reset=0;
$result = $db->prepare("SELECT charges AS charges,adm_date,discharge_date FROM admissions RIGHT OUTER JOIN wards ON wards.id=admissions.ward WHERE ipno='$search' AND discharged=:b  ORDER BY adm_date DESC LIMIT 1");
$result->bindParam(':b', $reset);
$result->execute();
for($i=0; $row = $result->fetch(); $i++){
    $bed_charge= ($row['charges']);
$discharge_date=$row['discharge_date'];
$ward_charge=$row['charges'];
if ($discharge_date="0000-00-00") {
$discharge_date=date("Y-m-d");
}
$adm_date=date("Y-m-d", strtotime($row['adm_date']));
$date1=date_create($adm_date);
$date2=date_create($discharge_date);
// calculate number of days
function dateDiff ($discharge_date, $adm_date) {
    // Return the number of days between the two dates:    
    return round(abs(strtotime($discharge_date) - strtotime($adm_date))/86400);

}
$days= round(abs(strtotime($discharge_date) - strtotime($adm_date))/86400);

if (isset($days)) {
    $bed_charge=$row['charges'];
    
?>
<table class="table table-bordered">
<th>date</th>
<th>description</th>
<th>cost</th>
<tbody>
    <?php
// Function to iterate between two dates and print them
function iterateBetweenDates($startDate, $endDate)
{
    //get ward charge
    
// Convert the string dates to DateTime objects
$start = new DateTime($startDate);
$end = new DateTime($endDate);

// Adjust the end date by adding one day, so it includes the end date as well
$end->modify('+1 day');

// Iterate through the date range
$interval = new DateInterval('P1D'); // 1 day interval
$dateRange = new DatePeriod($start, $interval, $end);

// Print the dates
foreach ($dateRange as $date) {

?>
<tr>
<td><?php echo  $date->format('d-m-Y'); ?></td>
<td>bed charges</td>
<td><?php global $bed_charge;  echo  $bed_charge; ?></td><?php } ?>
<?php
}
// Test the function
$startDate = $row['adm_date'];
$endDate = date('Y-m-d');


iterateBetweenDates($startDate, $endDate);
?>
</tr>
</tbody>
</table>
</tr>
<table class="table table-bordered" >
<thead class="bg-primary">
<tr>
<th>ward bed charges</th>
<th><?php echo $days." days";   ?></th>
<th>
<?php $admission_total=$days*$row['charges']; 
echo $admission_total;

}}
?>
</th>
</tr>
</thead> 
</table>