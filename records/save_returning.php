<?php
session_start();
include('../connect.php');
$a = $_POST['patient'];

$d = $_POST['name'];
$date = date("Y-m-d H:i;s");

//save data into visits table
$sql = "INSERT INTO visits (patient) VALUES (:a)";
$q = $db->prepare($sql);
$q->execute(array(':a'=>$a)); 
$j = $a;
$doctor =$_SESSION['SESS_FIRST_NAME'];
$date=date('Y-m-d');
//check if there is fees to be paid
if (isset($_POST['fees'])) {
$fees =($_POST['fees']);
$List = implode(', ',$fees);
$result = $db->prepare("SELECT GROUP_CONCAT(amount) AS amount FROM  fees   WHERE fees_id IN($List)");
$result->execute();
for($i=0; $row = $result->fetch(); $i++){
 $feeamount=$row['amount'];
$amounts = explode (",", $feeamount); 
foreach (array_combine($fees, $amounts) as $fee => $amount){

$sql = "INSERT INTO collection (fees_id,date,paid_by,amount) VALUES (:a,:b,:c,:d)";
$q = $db->prepare($sql);
$q->execute(array(':a'=>$fee,':b'=>$date,':c'=>$a,':d'=>$amount));

}
}
}
//end checking fees
//save lab requests
if (isset($_POST['lab'])) {
$labs = $_POST['lab'];
foreach ($labs as $lab) {
$sql = "INSERT INTO lab (test,opn,reqby) VALUES (:a,:b,:c)";
$q   = $db->prepare($sql);
$q->execute(array(
':a' => $lab,
':b' => $j,
':c' => $doctor
));
}
}
$served=1;
$has_bill=1;
$sql ="UPDATE patients
      SET  served=?,
           has_bill=?
      WHERE opno=?";
$q = $db->prepare($sql);
$q->execute(array($served,$has_bill,$a));
//

	header("location: receipt.php?name=$d&number=$a");
?>