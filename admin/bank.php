<?php 
require_once('../main/auth.php');
include ('../connect.php'); ?>
 <!DOCTYPE html>
<html>
<title>bank copy</title><meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0">
 <?php 
include "../header.php";
?>
  
</head>
<body><header class="header clearfix" style="background-color: #3786d6;">
        <?php include('../main/nav.php'); ?>   
  </header><?php include('sidee.php'); ?>   
<div class="jumbotron" style="background: #95CAFC;">
   <ol class="breadcrumb primary">
    <li style="float:right;"><?php
        $result = $db->prepare("SELECT status  FROM employees WHERE active=1 LIMIT 1");
        $result->execute();
  for($i=0; $row = $result->fetch(); $i++){ 
      $status = $row['status'];
      if ($status== 0) {
            # code...
             
      ?>
      <a href="activate_pay.php?status=1">activate payslips</a><?php } ?><?php
      if ($status==1) { ?>payslips activated<?php } ?><?php
      if ($status==2) { ?>salaries paid <a href="activate_pay.php?status=2">reset payslips</a><?php } ?><?php } ?></li>
   <li><a rel="facebox" href="add_employee.php">add employee</a></li> 
   <li><a rel="facebox" href="add_job_group.php"> add job group</a></li>
   <li><a rel="facebox" href="add_allowance.php">add allowance</a></li>
   <li><a href="bank.php">payroll bank copy</a></li>
   </ol>    
</nav>
<?php
if (isset($_GET['response'])) {
?>
<p class="alert alert-success">expense recorded</p>
<?php } ?>
<div class="container">
<label>generate payrol bank copy</label> 
<form action="bank.php" method="GET"><input type="text" id="mydate"  name="d1" autocomplete="off" placeholder="pick date" required/><button class="btn btn-success"><i class="icon icon-save icon-large"></i>submit</button></span>

</form>
</hr> 
<p>&nbsp;</p>
<?php
if (isset($_GET['d1'])) {
     ?>
     <div class="container" id="content">
<table class="table">
  <tr>
    <th>Period</th>
    <th>Account name</th>
    <th>Bank name</th>
    <th>Branch name</th>
    <th>Account Number</th>
    <th>Basic Salary</th>
    <th>Allowance</th>
    <th>Gross Pay</th>
    <th>NSSF</th>
    <th>NHIF</th>
    <th>PAYE</th>
    <th>Other deductions</th>
    <th>Net pay</th>
  </tr>
  <tr>
 <?php
 $d1=$_GET['d1']." 00:00:00";       
      $date1=date("Y-m-d H:i:s", strtotime($d1));
	   $period = date("Y-m");
	   $sql = "SELECT 
				CONCAT(YEAR(s.date),'-',RIGHT(CONCAT('0',MONTH(s.DATE)),2)) AS Period,
				e.employee_name AS `Account Name`, 
				e.bank AS `Bank Name`, 
				'' AS `Branch Name`, 
				e.account_number AS `Account Number`,
				j.basic_salary AS `Basic salary`,
				al.amount AS `Allowance`,
				s.gross_pay AS `Gross pay`,
				ns.amount AS NSSF,
				nh.amount AS NHIF,
				t.amount AS `PAYE`,
				od.amount AS `Other deductions`, 
				s.amount AS `Net Pay`
			FROM employees e
			INNER JOIN  `job_groups` j ON j.jg_id = e.jg_id
			INNER JOIN salaries_payments s ON s.employee_id = e.employee_id
			LEFT JOIN nhif_payments nh ON e.employee_id = nh.employee_id AND (CONCAT(YEAR(nh.date),'-',MONTH(nh.DATE)) = CONCAT(YEAR(s.date),'-',MONTH(s.DATE)))
			LEFT JOIN nssf_payable ns ON e.employee_id = ns.employee_id AND (CONCAT(YEAR(ns.date),'-',MONTH(ns.DATE)) = CONCAT(YEAR(s.date),'-',MONTH(s.DATE)))
			LEFT JOIN (SELECT SUM(amount) AS amount, employee_id, (CONCAT(YEAR(DATE),'-',MONTH(DATE))) AS period FROM allowance_payments GROUP BY employee_id, (CONCAT(YEAR(DATE),'-',MONTH(DATE)))
			) al ON e.employee_id = al.employee_id AND (al.period) = CONCAT(YEAR(s.date),'-',MONTH(s.DATE))
			LEFT JOIN tax_paid t ON e.employee_id = t.employee_id AND (CONCAT(YEAR(t.date),'-',MONTH(t.DATE)) = CONCAT(YEAR(s.date),'-',MONTH(s.DATE)))
			LEFT JOIN `other_deductions` od ON e.employee_id = od.employee_id  AND (CONCAT(YEAR(od.date),'-',MONTH(od.DATE)) = CONCAT(YEAR(s.date),'-',MONTH(s.DATE)))
			WHERE CONCAT(YEAR(s.date),'-',RIGHT(CONCAT('0',MONTH(s.DATE)),2)) = :a;";
	   
        $result = $db->prepare($sql);
		$result->bindParam(':a',$period);
        //$result->bindParam(':b',$date2);     
  $result->execute();
  $netpay = 0;
  $nssf = 0;
  $nhif =  0;
  $paye = 0;
  $other_deductions = 0;
  for($i=0; $row = $result->fetch(); $i++){ 
	  $netpay+=$row['Net Pay'];
	  $nssf+=$row['NSSF'];
	  $nhif+=$row['NHIF'];
	  $paye+=$row['PAYE'];
	  $other_deductions+=$row['Other deductions'];
   ?>
   <td><?php echo $row['Period']; ?></td>
   <td><?php echo $row['Account Name']; ?></td>
    <td><?php echo $row['Bank Name']; ?></td>
     <td><?php echo $row['Branch Name']; ?></td>
     <td><?php echo $row["Account Number"]; ?></td>
    <td><?php echo $row['Basic salary']; ?></td>
    <td><?php echo $row['Allowance']; ?></td>
    <td><?php echo $row['Gross pay']; ?></td>
    <td><?php echo $row['NSSF']; ?></td>
    <td><?php echo $row['NHIF']; ?></td>
    <td><?php echo $row['PAYE']; ?></td>
    <td><?php echo $row['Other deductions']; ?></td>
    <td><?php echo $row['Net Pay']; ?></td>
    </tr>
    <?php
    } ?>
	<tr>
		<td colspan="3"><b>Total</b></td>
		<td></td>
		<td></td>
		<td></td>
		<td></td>
		<td></td>
		<td><b><?PHP echo $nssf;?></b></td>
		<td><b><?PHP echo $nhif;?></b></td>
		<td><b><?PHP echo $paye;?></b></td>
		<td><b><?PHP echo $other_deductions;?></b></td>
		<td><b><?PHP echo $netpay;?></b></td>
	</tr>
   </table> 
 <?php } ?>
 </div>
 <script type="text/javascript">
   function printDiv(content) {
            //Get the HTML of div
            var divElements = document.getElementById(content).innerHTML;
            //Get the HTML of whole page
            var oldPage = document.body.innerHTML;

            //Reset the page's HTML with div's HTML only
            document.body.innerHTML = 
              "<html><head><title></title></head><body>" + 
              divElements + "</body>";

            //Print Page
            window.print();

            //Restore orignal HTML
            document.body.innerHTML = oldPage;          
        }


</script>
<?php if (isset ($_GET['d1'])){ ?>
      <button class="btn btn-success btn-large" style="margin-left: 45%;" value="content" id="goback" onclick="javascript:printDiv('content')" >print copy</button>
      <?php } ?>
 <script>
  $( function() {
    $( "#mydate" ).datepicker({
      changeMonth: true,
      changeYear: true
    });
  } );

  </script>
  <script>
  $( function() {
    $( "#mydat" ).datepicker({
      changeMonth: true,
      changeYear: true
    });
  } );
  </script> 

</div>
</body>
</html>